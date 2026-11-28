# HANDOFF — the refav1 LONGITUDINAL stream

*Rewritten 2026-09-05T22:45Z, after `lonshift` landed and after two of my own
errors were caught and fixed. Read `RESULT.md` first — §6 is the result, §6.5
prices the levers I declined, §6.6 is the error. Every queued arm's BOTH
outcomes are committed in §5 before its numbers exist.*

## 1. THE STATE IN ONE TABLE

| arm | status | what it says |
|---|---|---|
| `wk15` | banked (baseline) | `ccos`, `(0.0, 15.11245, 64.297)`, seed 0 — every comparison is against this |
| `wk151` | banked | the **seed floor**: `wk151 − wk15` on `LON_speed_mae` is **+0.0076 [−0.0134, +0.0282]** |
| **`lonshift`** | **LANDED — the result** | D2 (`--a-sustain-mode a0_shift`). LON family separated at **29.8× / 8.2× / 52×** the floor; beats `ha0_ext` on **FDE** (−0.3723, separated); cuts the LON gap **47 %**; **still loses LON (+0.2599)** |
| `lonseam` | ⛔ **VOID — inert by construction** | `--jerk-seam a0` under `W_JERK = 0.0`. Arithmetic, not a null. **Do not quote it.** Guarded now |
| `lonshift_s1` | queued #1 | the **replicate D2 owes** — see §3.1 |
| `seambase` / `seamon` | queued #2, #3 | the seam, done properly: both at `W_JERK = 0.02`, seam the only variable |
| `lonvocab` | queued #4 | D1, for attribution (was maintain-only enough?) |
| `loncomb3` | queued #5 | both levers on a triple where **both can act** |

## 2. One command to land whatever has arrived

```
bash <scratchpad>/finalize_lon.sh
```
Idempotent, zero GPU, and it **names the arms whose record is absent** rather
than silently thinning the panel. ⚠️ Its `NEW` list still names the v2 arm names
— update it to `lonshift_s1 seambase seamon lonvocab loncomb3` before running,
or add them.

Per-arm, the two readings that matter are:
`lon_emitted.py <arms>` — **did it act?** — and `refav1_paired_delta.py` for the
four families against `wk15`, `ha0` and `ha0_ext` with the arm-against-itself
control that must read `+0.0000 [0, 0]`.

## 3. THE BAR — stated before the numbers

1. ⛔ **A separated CI is NECESSARY AND NOT SUFFICIENT.** Proven live this turn:
   `lonshift − wk15` reads `TAC_traj_lat_correct` **−0.1000 [−0.1750, −0.0250]**,
   *separated* — and the **seed pair reads the same interval bit-for-bit**, so it
   is **not attributable to the lever**. Always read the lever's delta against
   the seed pair's delta **on the same metric**.
2. **Four families or it is incomplete**, per-family, never pooled.
3. **"Beats the floor" must be conjoined with "acts".** Report `frac a ≡ 0`,
   `mean|a|` and `a[0] == a_goal[0]` beside every ADE. `lonshift` moved
   `frac a ≡ 0` **0.475 → 0.000**; that is what makes its win a driving result
   rather than a tie reached by stopping.
4. **Every arm carries its cost triple, its vocabulary mode and its seed.**
   ⚠️ The triple's **third entry is a DEAD TERM** (`target_speed` is never
   passed; `test_C1` pins the call site) — quote it, never attribute to it.
5. ⛔ **The expressivity tables (§1.5, §1.9, §6.5) are NOT planner results**, and
   they **over-predict by a measured 1.71×** — D2's row predicted +0.1517 and the
   landed arm realised +0.2599. Use that discount on any new candidate before
   spending compute; it is what declined D4.

### 3.1 What `lonshift`'s result still owes

**`lonshift_s1` (`--plan-seed 1`, everything else identical).** The floor quoted
in §6.2 is **`wk15`'s** seed pair, not `lonshift`'s own. At **29.8×** the effect
is very unlikely to be noise — but the formal claim needs the replicate, and
until it lands the caveat travels with every quotation of the number.

## 4. Levers, with the ones already closed marked

| lever | status |
|---|---|
| D2 `a_shift` (all relative targets) | ⭐ **LANDED, the win** |
| D1 `a_sustain` (maintain only) | queued, for attribution |
| jerk seam | queued **properly** (`seambase`/`seamon`) after the inert arm |
| **D4 raise `GOAL_A_MAX`** | ⛔ **MEASURED AND DECLINED** — 13.6 % expressivity, ≈4.6 % realised after the 1.71× discount; and bounded, since clip 2.5 ≡ clip 4.0 (`max\|a0\|` = 2.31) |
| **D5 shorten `GOAL_REACH_S`** | ⛔ **REFUTED** — worse than D2 on every headline column; the τ = 2.0 control reproduces D2 to four decimals, so the negative is a measurement |
| arm `W_VEND` | ⛔ **PI DECISION** — `test_C1` pins the call site, `test_C2`'s docstring reserves it verbatim. Motivating measurement in §1.2–§1.4 |
| a LON level set + trained chooser | last. The current design deliberately has **no free parameter**; adding one re-opens the oracle-vs-realised problem |

⇒ After D2 the residual is **broad** (84.3 % at `v0 ≥ 2 m/s`), both vocabulary
knobs are closed, so **the next lever is the COST side** — which is what
`seambase`/`seamon` finally test properly.

## 5. ⛔ THE FOUR TRAPS THIS TURN PAID FOR — do not re-pay them

1. **A lever multiplied by zero is not a null about the lever.** `lonseam` ran
   `--jerk-seam a0` under `W_JERK = 0.0` and banked `+0.0000 [0,0]` on all ten
   metrics. `refav1_arm.py` now **refuses** that combination before the rollout;
   both directions of the guard are exercised. **Before any arm, assert every
   lever it names is multiplied by a LIVE coefficient in the config it will
   actually use.**
2. **A gate must count the ARM's artifact, not any artifact.** Counting distinct
   `--out` targets fixed M28's parent/child process trap, but `--out` is not
   unique to the arm tool — a sibling's analysis script (`--out
   ../raw/l3_splitp30k.json`) was counted as an arm and the gate could never
   open. Require the **tool name** too.
3. **A process search self-matches.** Building a kill pattern that appears in
   your own command line kills your own shell (exit 255, measured). Assemble the
   pattern from disjoint pieces and verify zero against a control that must read
   non-zero.
4. **A red test seen during your own multi-step edit is evidence about your
   edit.** I attributed `test_A6` to a sibling stream; it does not reproduce
   (26 / 70 / 647 passed). Re-run from a **quiescent tree** before attributing
   anything to anyone else.

## 6. ⚠️ THE OPEN COORDINATION RISK — name it, do not work around it

**My queue is gated at 2 concurrent arms and the sibling stream is launching
arms back-to-back**, so both slots have been continuously occupied
(`combined`, `wk15_ladder`, `best`, `bestlad`, `combined_seed1`, `ta_wk15_s0`,
`ta_wk15_s1` …). `lonshift_s1` has not started. The gate is behaving correctly —
a third arm is a measured OOM risk on the 8 GB 4060 and would endanger the
sibling's arms as well as mine — so this is **not** something to solve by
raising the cap.

⇒ **This needs a scheduling decision at the Master-Mind level**, not a local
workaround: either an explicit slot share between the two refav1 streams, or an
agreed priority for `lonshift_s1`, which is the one arm that converts a landed
result into a quotable one.

## 7. Not this agent's to decide

* **`a_shift` / `a_sustain` / `jerk_seam_a0` becoming refav1's DEFAULTS.** All
  implemented, pinned, OFF by default, bit-identical when off. These arms
  **measure** them; they do not authorise them.
* **Arming `W_VEND`** — §4.
