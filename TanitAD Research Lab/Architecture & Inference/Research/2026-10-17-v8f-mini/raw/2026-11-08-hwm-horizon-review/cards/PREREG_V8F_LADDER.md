# PREREG_V8F_LADDER: does a world model trained on the PI's bands predict inside them? (Master Mind, 2026-11-08, written before any LADDER code or run)

## Why
The PI's definition (`theory/horizon_review/REQ_HORIZONS.md`): operative up to 2 s, tactical up to 6 s, strategic 6 to 30 s. The world model trains one step at 0.2 / 0.4 / 0.8 s (`AUDIT_MODEL.md` F1), so no level is trained inside its band. IMAG1B (`40500d68`) found no level predicts better than copying the present. This card trains the ladder of `SPEC_HWM.md` section 2 and asks whether each level beats copying **inside its own band**. MRATE's multi-rate history is folded in as an arm, because the frozen MRATE arms sit on the rejected ladder.

## Arms (world-model only: adapter, readers and planner frozen at M2b s{k}, as MRATE. Driving cannot change, so this is T0.)
| arm | strides | steps (s) | trained k (L1, L2, L3) | history |
|---|---|---|---|---|
| **C0 (control)** | (2,2) | 0.2 / 0.4 / 0.8 | 1, 1, 1 | as M2b |
| **S35 (step only)** | (3,5) | 0.2 / 0.6 / 3.0 | 1, 1, 1 | as M2b |
| **R35 (step and reach)** | (3,5) | 0.2 / 0.6 / 3.0 | 10, 10, 6 | as M2b |
| **R35H (and multi-rate history)** | (3,5) | 0.2 / 0.6 / 3.0 | 10, 10, 6 | K = 3 at each level's own step |

* L3 steps with no frame available are masked and counted. The same windows, data, optimizer and step count are used across arms, except where the card states otherwise. Seeds 0 and 1.
* **R35b, the replicate:** R35's flags and seed 0 again, zero levers moved, for the run-to-run floor (`H-ESTIM-SEED-1`).
* Every lever is named in each argv, read back from the BUILT model, and each run passes the RECOVER-NEXT selfcheck with a per-run token.

## Read (T0, IMAG1B's instruments and rules, with horizons extended; both training seeds required)
Latent targets are encoded future frames. Each level is read at its own reach:
* L1 at 1.0 and 2.0 s; L2 at 3.0 and 6.0 s; L3 at 6, 9, 12 s (and 15, 18 s beside, because few windows have them).
* **Deciding per level:** IA-0 skill against Copy-Last in the level's latent at its band end (L1 2.0 s, L2 6.0 s, L3 12 s), by IMAG1B's two-part rule, and IA-0 health.
* **Beside, no bar:** IA-1 capture, IA-2 probe transfer (the speed, lead-gap and yaw-rate targets), the IA-3 action audit, TSI and the cross-level slopes, and PLANTEST (`PREREG_V8F_PLANTEST.md`) on the same instrument.
* The T0 caveat applies: nothing here is driving performance.

## Committed outcomes (per arm, both seeds)
* **BAND-MET:** all three levels beat Copy-Last at their band end, and are HEALTHY at every level.
* **BAND-PARTIAL:** at least one level does, the rest do not. The levels are named.
* **BAND-MISS:** no level does.
* A difference between two arms counts as a lever effect only if it exceeds the R35b replicate floor.

## What each outcome triggers (committed now, Rule Zero)
* **BAND-MET** in R35 or R35H: the arm with the smaller reading cost becomes the world model of record, and HWM-PLAN (`SPEC_HWM.md` section 4) runs on it.
* **BAND-PARTIAL:** the passing levels are used, and the failing level gets the next lever by its read: lateral or action failure goes to the action-diverse sampler, re-binned on the level's band; a missing scene target goes to the token-format result of CAPLADDER; a healthy but copying level goes to a higher inverse-dynamics weight.
* **BAND-MISS:** the same diagnosis at every level, and the token format moves ahead of a second ladder run.

## Controls (the read is INVALID if one fails)
1. **k = 1 identity:** with k = 1 the new loss equals the old loss bit for bit on a fixed batch.
2. **Reach reachability:** gradient reaches each level's predictor through every rollout step in use. A mutation that deletes the k-step line goes red, as does a mutation that leaves the strides unapplied.
3. **Mask check:** a step past the frames that exist contributes exactly 0.0, and a step inside contributes non-zero.
4. **Planner identity** as MRATE control 1, on a fixed batch after the run.
5. IMAG1B's controls 2, 3 and 4 on the read.

## Implementation choices
The implementer writes `theory/ladder/IMPL_CHOICES_LADDER.md` (multi-step loss weights and schedule, window sampler and pooled path, the mask rule, the K = 3 history definition, the argvs, the projected box-2 time per run) and records its sha256 before the first run. If the projected total is over 72 h of box-2 time, the Master Mind amends this log before any run.

## Log
* 2026-11-08 10:24:26 frozen: prefix 4442 bytes, sha256 5f944c98fda7e8a2e7d14c93e484bdd2903458f119ccad21f39351bab93ebf3c (everything above this line). No LADDER code or run existed at this time.
