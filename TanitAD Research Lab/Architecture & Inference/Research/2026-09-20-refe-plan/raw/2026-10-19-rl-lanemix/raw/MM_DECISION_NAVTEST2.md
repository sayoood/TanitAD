# RL-2: Master Mind decision on navtest look #2 (2026-10-18, BEFORE V1's navtest outcome exists)

**Decision: DEFERRED, and made CONDITIONAL on RL-1 V1's navtest outcome.** That outcome is not known yet: V1's scoring run is waiting for memory, `rl/nav/nt_run2.py`. The rule below is fixed now, so the choice cannot follow the number.

**Why not spend the look now.**
- On the HELD half, L0.25 cannot be distinguished from V1: contrast **+0.023 [-0.150, +0.200]**.
- V1 itself passes that half: +0.285 [+0.018, +0.548].
- RL-val cannot see the navtest TTC/NC loss that motivates the lane family, so the RL-val read cannot rank L0.25 above V1.
- A second look on a variant the selection tier cannot separate from the first would add a forking path, not information.

| V1 navtest outcome (PREREG_RL1_B section 5) | look #2 for L0.25 | reason |
|---|---|---|
| **N-PASS** | **NO** | V1 is the composite; L0.25 is not separable from it on the selection tier |
| **N-GAIN-LOST** (safety held) | **YES**: a NEW registration, its own bar (the RL-1 bar), seam via `rl/nt_build.py`, ~40 min CPU | L0.25 restores 0.25 of RL's lane delta, the carrier of the gain (`mechanism_r0*.json`). This is the one outcome where more lane is the indicated direction |
| **N-SAFETY-LOST** or **N-BOTH** | **NO** | L0.25 carries MORE RL lane than V1. If V1 already loses NC/TTC on navtest, more lane is the wrong direction. The next lever is the anchored RL arm (RL-1 section 7), BLOCKED on A40 compute: P6 holds the A40, and the PI declined extra compute |
| INVALID (gates fail) | NO decision | fix the run first |

**Disclosure:** if look #2 is spent, every report quoting it states that it is the **second** navtest look in the RL head-wise family, and quotes V1's look beside it.
