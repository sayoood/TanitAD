# TanitAD program report — 2026-08-28 evening (D-025, late)

*Master Mind · 22:12 Europe/Berlin (measured, Thor clock) · branch
`agent/arch-inf-20260803` · ⚠️ the 12:57 and 17:57 slots were MISSED — the CLI
process was suspended ~10:45–22:10; nothing moved in the gap (HEAD unchanged,
index 0 staged, Thor idle, no trainer process) and no work was lost.*

## 1. What closed since the morning report (all committed before the gap)

1. **MM-E1 (P0 EMA bake-off): 🔶 MIXED** (`552c450d7`, T0-DIAGNOSTIC, MEASURED,
   config-diff CLEAN both arms):

   | read | two-term base | ema2k_s0 | ema2k_s1 | seed spread |
   |---|---|---|---|---|
   | drift r | 0.4531 | **0.4002** | **0.3644** | 0.0358 |
   | cos (centred) | 0.1845 | 0.1336 | 0.0979 | 0.0357 |
   | nrmse | 0.9876 | 0.9912 | 0.9954 | 0.0042 |

   Drift BETTER beyond spread on both seeds — ⭐ **the first measured lever that
   moves drift down on the trainable line at any scale** — but cos WORSE beyond
   spread, so per the committed table: numbers stated, **no verdict** (C160).
   The warmup-transient hypothesis (EMA teacher lags most at 2k) is on file;
   only a 30k pair discriminates it. Participation was not emitted by either
   probe — recorded as an instrument gap.
2. **The v7 ground truth advanced under the PI's four-point pass** and
   **D-LABEL-GT re-pinned same turn** (`3e1af666a` + `9986f6710`): blob
   `4dd31d33`, 0 exclusion violations, contradictions 0.000 %, the
   `corroboration` tiers replacing the erroneous "~13 % grounding" (DE-C150:
   41.1 % checkable / 70.1 % either-source), and the scene block pinned as
   ⛔ presence-never-reaction (the 710-clip conversion refuted by base-rate
   control).
3. **LAB-RUN-002 landed and routed** (`27021d506` + `c4dba4fc2` + `8713b9e17`):
   4/4 domains, 18 primaries banked (library → 116). New register row
   ⛔ **D-B1-GATE** — the TRT sm_110 silent-FP32 fallback makes any naive
   "INT8 validation" a false positive; B1 blocks until the hardened gate
   protocol lands (owner DeployFlyWheel). EvalFlyWheel got a durable brief
   (its session is down): pin **navhard-two-stage EPDMS** — two "EPDMS"
   protocols sit ~30 points apart and our Drive-JEPA reference has NO navhard
   number.
4. **Zero stranded work programme-wide** (`dd0f519e7`): both worktrees drained,
   the last orphaned staging banked, shared index at 0.

## 2. Fleet at 22:12 (MEASURED)

Thor idle (no trainer; o14fut30k + both P0 arms banked, EMA trainer installed
with backup) · dev box idle · no FlyWheel session traffic since ~10:40 ·
13 commits today on the agent branch.

## 3. The decision queue for Sayed (unchanged since morning, defaults stand)

1. **NavSim split** — default: `navhard_two_stage` only (31 GB); no download
   without your go.
2. **Untimed CoT supervision** — yours by directive; the Lab's two-option
   material (MIL vs alignability gating) is ready.
3. **B1 signal** — now additionally gated by ⛔ D-B1-GATE (the hardened
   quantisation gate must land first).
4. **EMA 30k pair** — default: run after the v7r launch decision (Thor idle;
   ~8 h; it answers whether the first-ever drift reduction survives scale and
   whether the cos cost was warmup transient).
5. **The two-arm NavSim entry design** (benchmark-standard + vision-pure) —
   proposed by the Lab in lieu of a doctrine waiver.
6. **CORRIDOR_OFFSET lane-detector reference arm** — PI-gated by agreement
   with the DataFlyWheel; chance-baseline requirement specced.

## 4. Where v7 stands tonight

The recipe (`V7_RECIPE_AND_SCALEUP.md` §5.1) now carries: two-term core +
SIGReg + `omega_accel_v` + **O14-fut (R2, ABSORBED — anti-displacement)**, on
the trainable encoder with DINOv3 distill-init and the v7 vocabulary wired
end-to-end. The single open scientific front is the **drift attractor** —
now cleanly separated from displacement, with the EMA teacher as the first
lever ever measured to move it (2k, MIXED). L4 (T1 driving) remains the next
proof rung and is compute-ready the moment you green-light the scaled run.
