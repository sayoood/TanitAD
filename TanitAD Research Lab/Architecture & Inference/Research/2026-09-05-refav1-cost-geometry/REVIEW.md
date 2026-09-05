# REVIEW — `2026-09-05-refav1-cost-geometry`, run against the programme's own rules

*Self-review under `TanitAD_Review`, 2026-09-05, before the package is quoted
onward. Every headline number in `RESULT.md` was re-read from the RECORDS
(`rec_*.json`), never from the prose. Purpose is to certify assets, so the sound
list is as much of the output as the findings.*

---

## FINDING 1 — ⛔ **WITHDRAWN CLAIM (mine): "the seed ladder is not free"** · SEVERITY: HIGH · FIXED IN THIS TURN

**Rule violated:** the package's own §4 sufficiency rule — *a difference is a lever
effect only if it exceeds the seed pair's difference **on the same metric***.

**The claim.** §7.7 and the `D-REFAV1-CG-L3-NULL` row asserted *"the ladder is not
free: `lane_keep` recall 0.7143 → 0.5714 (~2× the 0.0750 floor) and goal FDE
2.9639 → 3.3200."*

**The evidence against it.** `ccos_seed1` — same flags, only `--plan-seed` differs —
reproduces the identical movement with **no lever at all**:

| tactical metric | `ccos_argmax` | seed replicate (NO lever) | `l3ladder` | verdict |
|---|---|---|---|---|
| `lane_keep` recall | 0.7143 | **0.5714** (floor 0.1429) | **0.5714** (delta 0.1429) | inside the floor |
| goal FDE m | 2.9639 | 3.3079 (floor 0.3440) | 3.3200 (delta 0.3561) | inside the floor |
| TAC lateral kappa | 0.3795 | 0.2822 (floor 0.0973) | 0.2822 (delta 0.0973) | inside the floor |

**Root cause: THE WRONG FLOOR.** 0.0750 is the *paired* `TAC_traj_lat_correct`
floor; the per-class `lane_keep` recall's own seed variation is **0.1429**. A
cousin statistic was used as the floor for a different one — the **statistic
ambiguity** class of the taxonomy, in the very rule meant to prevent it.

**Fix (applied):** `RESULT.md` §7.7 and the register row both carry the withdrawal,
the table above, and the corrected reading — **the ladder is inert on all four
families, neither costly nor beneficial**, which makes the L3 null *total* rather
than weaker. The `--seed-kappa-ladder` docstring's warning ("with `W_KAPPA = 0` a
wider curvature set can only add ways to be wrong") is therefore recorded as
**UNTESTED on this panel, not vindicated**.

---

## FINDING 2 — ⛔ **RETRACTED BEFORE ASSERTION: "the lever fields are missing from the records"** · SEVERITY: none (my probe was wrong)

A first probe read `d["refav1"]` and reported `kamm_mu` / `seed_kappa_ladder` /
`goal_kappa_turn` as `<<MISSING KEY>>` on **every** arm — which would have made
`kamm07`, `combined` and `l3ladder` unquotable under the brief's own rule. A second
probe, searching the whole record rather than one dict, found them at
`refav1.manifest.plan_cfg.kamm_mu` and `refav1.manifest.goal_rule.seed_kappa_ladder`.

**Record provenance is COMPLETE and CORRECT:**

| arm | `seed_kappa_ladder` | `kamm_mu` | metric | `W_KAPPA` |
|---|---|---|---|---|
| `l3ladder` | `[0.002, 0.005, 0.01, 0.02, 0.04]` | `None` | ccos | 0.0 |
| `kamm07` | `None` | **0.7** | ccos | 0.0 |
| `combined` | `[0.002, 0.005, 0.01, 0.02, 0.04]` | **0.7** | ccos | 0.0 |
| `ccos_argmax`, `wk15`, `cos_wk` | *(key absent — predate the field)* | — | — | 0 / 15.11245 / 0.05 |

**Class: absence at one location.** Logged because the near-miss is the lesson —
the same class this package already retracted once today (§6.1).

---

## WHAT IS SOUND — certified

1. **Quotable-number audit: PASSES.** Every headline figure in §7.1–§7.8 was
   re-read from `rec_*.json` and matched — ADE and its interval, curvature /
   heading / yaw-rate / cross-track, speed and accel MAE, tactical kappa, the three
   per-class recalls, goal FDE. No number in the report comes from prose.
2. **Estimator discipline: PASSES, mechanically.** `overlapping_holdout_se` appears
   **0** times in `RESULT.md` (control: "episode-cluster" appears **5**); **90**
   estimator blocks across all records are `episode_cluster_bootstrap` /
   `paired_episode_cluster_bootstrap`, **0** deprecated.
3. **Tier stamps: PASSES.** **0** arms missing a tier. T0 arms (`ol`,
   `cl_oraclegoal`, `cl_oracleseed`) are labelled T0 and are never called driving
   performance.
4. **Panel scope: PASSES.** All records read **(40 windows, 8 episodes)** — a
   single panel, no mixing. `RESULT.md` states it in the header and the
   `MODEL_REGISTRY.md` row states its denominator in its first sentence,
   explicitly against the 282-window v7.2 EVAL grid.
5. **Controls: PASSES, and they did work.** A ground-truth control that **failed
   first** and voided a whole block (`vmin = 0`, §6.2); a must-read-non-zero
   control beside every "exactly 0.0" (`wk15` at 1.508e+01 m against the `ccosh`
   null; three non-zero arms beside `combined`'s `kamm_over 0.0000`); an
   arm-against-itself known-value control at `+0.0000 [0,0]`; a seed replicate used
   as the sufficiency floor throughout.
6. **Pre-registration: PASSES.** `PREREG_COST_GEOMETRY.md` carries **both outcomes**
   before any arm ran, plus two later addenda that are timestamped and state what
   had *not* yet been observed. Two of its predictions were **refuted and reported
   as written** (P1, and the shipped-arm prediction that turned out stronger than
   registered).
7. **Register: PASSES.** Every asserted and refuted claim landed in
   `GOALS_AND_CLAIMS.md` in the same turn (13 `D-REFAV1-CG-*` rows), and the
   `MODEL_REGISTRY.md` row was added the same day.
8. **Banking: PASSES.** All ten arm records and dumps are under `raw/arms/`
   (~110 files); nothing load-bearing lives only on the dev box. The rescued
   instrument `taniteval/tools/refav1_paired_delta.py` is in the repo.

## Residual work items (named, not fixed here)

* `kamm07`'s residual `kamm_over 0.1481` is a **scope** difference (cap on
  candidate controls vs `assert_feasible` on the recovered path) — stated in §7.6,
  closed empirically by `combined`, but the two objects are still not reconciled.
* Turn recalls are n = 8–11 and **underpowered**; `wk15`'s left/right asymmetry is
  flagged as a work item, not a finding.
* No seed replicate exists for any arm except `ccos_argmax`. Every lever verdict
  here rests on a single inference seed plus a floor measured on one pair.
