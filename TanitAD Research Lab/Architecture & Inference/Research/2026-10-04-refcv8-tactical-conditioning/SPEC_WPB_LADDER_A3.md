# SPEC_WPB_LADDER — AMENDMENT A3 (Master Mind, 2026-10-11 ~22:00 Europe/Berlin)

Registered before any ladder arm has a scored number. It follows the PI's two rulings of 2026-10-11 on refcv8's inputs
(`Project Steering/SPEC_REFCV8_DRAFT.md`, REGISTERED sha256 `90d36359…`, amendment A1) and RELAUNCHES the ladder on them: the
running ladder was built on the past-only speed input N2 and the route checkpoint RC-A50, and refcv8 will carry neither.
Every clause of `SPEC_WPB_LADDER.md` (sha256 `e1b6aff7…`), A1 (`cbd64fa2…`) and A2 (`8c7c73bb…`) applies unchanged — the
rig, the seeds, N, the levers of L2 and L3, the floor F, the bars, the reading order, the gradient-share schedule — except
the INPUT clauses below.

## Why (the PI, verbatim, 2026-10-11, in chat; spelling kept)
* **Route:** *"just give the nav command and the distance."* ⇒ no route checkpoint (`--r8-no-rc`); the route input is the
  per-frame nav command and its turn distance.
* **Speed:** *"yes use the quantified lbel as input, it is not a training signal, it the maximal wished speed. it is ok that
  it is matching the future speed in some cases since it also must be achieved in the future. It is important to assure the
  road type quantization and avoid the exacvt value of the future speed as target speed."* ⇒ the speed input is refcv7's
  4-way ROAD-TYPE set-speed sidecar {30, 50, 100, 120} km/h, selected by `--r8-speed-input label`, with the trained unknown
  row kept (`--r8-speed-unknown-p 0.45`). The trainer refuses any sidecar value outside the four.
* **Scope:** *"You can stop the ladder after 11 runs."* (unchanged; SPEC_REFCV8 §10.0).

## The amendment
1. **Speed input, EVERY arm, V0 included** (replaces the base SPEC §1 "Speed input (MM ruling Q5)" bullet): the road-type
   label. `--r8-speed-input label` replaces `--r8-speed-input n2`; `--speed-max-sidecar-v6` and `--speed-max-sidecar-v6-eval`
   are carried on every arm, the same two files on every arm; `--r8-speed-unknown-p 0.45` is unchanged. Because every arm
   carries the same channel, L1–L3 do not move it — as before. The v8-sidecar refusal of the base SPEC is SUPERSEDED.
2. **Route input** (replaces the base SPEC §1 "Labels" bullet's RC clause): no route checkpoint. The L1 bundle (`R8_BUNDLE`)
   drops `--r8-rc-variant`, `--r8-rc-noise-along-m`, `--r8-rc-noise-lat-m` and `--r8-rc-dropout`, and carries `--r8-no-rc`.
   `--r8-nav-from-v9` and `--r8-nav-args-dropout 0.5` are unchanged: the nav command and its turn distance are the route
   input.
3. **Relaunch.** The ladder restarts from V0 on the ruled inputs. V0's partial run on the old inputs, and its single
   gradient-share reading at row 500, are set aside — moved, never deleted — and no number from them is read, quoted or used
   for the §4.2 sizing. The §4.2 k is sized on the median of the ONE reading at row 500 of the relaunched V-R8 (A2 item 2).
4. **The 11 arms** (content on the ruled inputs; delta = the one variable of each arm against its base):

| arm | = | role |
|---|---|---|
| **V0** | refcv7 recipe at the rig, v7.2 labels, `--r8-speed-input label` + both sidecars + unknown row | L1 base |
| **V-R8** | V0 + `--refcv8` + `--r8-nav-from-v9` + `--r8-no-rc` + nav-args dropout 0.5 + the weights of the base SPEC §2 + the 693-box mask | L1 treatment; the L2 / L3 base; the §4.2 sizing substrate |
| **V-R8d** | V-R8 + `--r8-derange-feed` ONLY | L1 deliberate regression |
| **V0r**, **V-R8r** | V0 / V-R8 at training seed 1 | L1 replicate floors |
| **V-TACk**, **V-TACk-roll**, **V-TACkr** | as the base SPEC §2 | L2 |
| **V-MAP4**, **V-MAP4-roll**, **V-MAP4r** | as the base SPEC §2 | L3 |

   **Not run (PI scope):** L4 (V-VSHUF) and its hypothesis H-R8-X3-1, L4b (V-R8-E8, V-R8-E8-roll), L5 (V-R8-DRV,
   V-R8-DRV-roll). With them go the base SPEC's 12th arm and A1's four arms.
5. **V-R8d's new content.** The `--r8-rc-roll` half (each row gets another window's route checkpoint) has nothing to roll and
   is DROPPED; the arm is `--r8-derange-feed` only (the planner reads ANOTHER window's tactical output in training). The
   trainer refuses `--r8-rc-roll` without an RC (base SPEC §6 P3), so leaving it in the argv would refuse the arm, not run it.
6. **Eval-only rows** (replaces the base SPEC §2 list): **LEGAL** = the speed `unknown` row, no nav args
   (`nav_args_known = 0`), token kept. The **RC-OFF**, **RC-shuffled** and **E-1 floor** rows are NOT APPLICABLE (no RC).
   VMAX-OFF (the unknown row everywhere) and VMAX-SHUF (the label taken from a deranged window) read V-R8's X3 rows as in
   SPEC_REFCV8 §8.3.
7. **Stamps** (replaces the base SPEC §9 RC lines): *no route checkpoint (PI 2026-10-11); nav and turn distance are
   ego-future derived, so optimistic on PhysicalAI; speed input = the road-type label (PI 2026-10-11), every speed number
   quoted beside VMAX-OFF and UNKNOWN.*

## Pre-registered reading of B-REG under the new V-R8d (committed now, both outcomes)
B-REG (base SPEC §5) is applied AS WRITTEN: "V-R8d's turn direction-correct gain over V0 is ≤ F, and V-R8 − V-R8d is
separated with R > 1." What changed is what V-R8d breaks. Before A3 it deranged the tactical feed AND the route checkpoint,
the only route-bearing feed other than the nav token. Now it deranges the tactical feed only; the v9 nav token and turn
distance (`--r8-nav-from-v9`) are fed to V-R8 and V-R8d alike. Whether the nav command and its distance alone carry a
turn-direction gain over V0 is UNMEASURED (no arm on the ruled inputs has a number).
* **V-R8d gain ≤ F and V-R8 − V-R8d separated with R > 1** ⇒ B-REG PASSES; L1 is quotable as written.
* **V-R8d gain > F** ⇒ B-REG FAILS and L1 is NOT QUOTABLE (base SPEC §5: otherwise the instrument is broken and nothing
  from the rung is quotable). The reading is not rescued by arguing that the nav feed carried the gain. The next lever is a
  regression arm that ALSO deranges the nav feed; none of `--r8-derange-feed`, `--r8-rc-roll`, `--r8-roll-targets {tac,map}`
  or `--r8-roll-speed-input` rolls the nav, so it needs a new trainer flag — an UNREGISTERED proposal, not an arm.

## What is lost (disclosed; nothing scored)
* V0's partial training on the old inputs and its one row-500 gradient-share reading: set aside. The wall-clock it consumed
  is not recorded here.
* The eval-only RC rows and the E-1 floor on every ladder arm (no RC to read).
* The N2 channel on the ladder. Its only dedicated test, L4, was already out of scope.

## What is unchanged
Seeds (0; replicates 1), N = 3,000, the rig, the eval rows and sampler seeds, the floor F, every bar and the reading order
(base SPEC §5, A1's B-E8 / B-DRV are moot with their rungs), pose-sync on every arm, and A2 in full: `--grad-share-terms
tac_v6,map_hires`, `--grad-share-at gs_at(N)` = rows 500, 2000, 2500, `CAP_S` 3600 with `GS_S` 1,100 s, and `fan_min_vmax` stored.
The one-variable audit still applies to the ACTUAL argv of every arm.

## Code that must match this text (OWED — not done by this amendment)
`code/ladder/ladder_arms.py` at the tip (lines read 2026-10-11): `REMOVED` (L37) must stop stripping the two sidecars from
the refcv7 argv; `base_v0` (L95) sets `--r8-speed-input` to `label` instead of `n2`; `R8_BUNDLE` (L43–L44) drops the four
`--r8-rc-*` entries and gains `("--r8-no-rc", [])`; V-R8d (L112–L113) becomes `[("--r8-derange-feed", [])]`; the audit's
`no_v8_sidecar_anywhere` assertion must invert to "both sidecars on every arm, same paths". `ORDER` queues the 11 arms. The
build of `--r8-speed-input label` is with the stream that owns it (`hold_r8label`). The one-variable audit must PASS on
the rebuilt argv of every arm BEFORE the relaunch.
