# A6 diagnosis: where the A5 floor's harm lives (descriptive, computed after the A5 read, never deciding)

The inputs are the banked A5 rollouts (`raw/2026-11-06-levelfloor-a5-lf5-result/`) and their tick dumps on the scratch disk, all on
eval-141 `reindexed`. Per-window differences against RIC0 are summed inside window classes and divided by the reader's n, so the
class contributions add up to the reader's delta. Scripts and JSON are in this folder.

1. **The heads' harm is overshoot** (`f_harm_split.py`, `F_HARM_SPLIT.json`). In windows where the floor speed exceeded the logged
   speed by more than 0.25 m/s on a bound tick, ADE6 is +1.088 of +1.089 m (RILF100_s1).
2. **Most closure is preventive** (`f_harm_xtab.py`, `F_HARM_XTAB.json`).
   * Only 0.113 / 0.106 (exact level) and 0.211 / 0.208 (heads) of the bound ticks have v0 < 5.2 m/s.
   * Windows with a bound tick at v0 >= 5.2 carry −0.0317 of the exact arm's −0.0394 M1 effect (s0).
   * A gate on the ego's own speed would therefore drop most of the closure.
3. **The heads as a binary classifier are weak where it matters** (`heads_binary.py`, `HEADS_BINARY.json`).
   * On M1 ticks, v_hat >= 5.2 holds on 0.572 / 0.458.
   * On ticks with a logged speed below 3 m/s, v_hat >= 5.2 holds on 0.221 / 0.186.
4. **With the exact level, the harm sits where the floor HOLDS speed against O's slowdown** (`hold_split.py`, `HOLD_SPLIT.json`).
   * Windows where F asked for more than max(v0, 5.5 m/s) are where it helps: ADE6 −0.1975 / −0.1840.
   * The other bound windows carry all of DK-VIOL (+0.0235 / +0.0135) and of cross-track (+0.1108 / +0.1073).

   **Reading:** the floor holds the first second's level constant over the 6 s plan while the human slows later in it.
5. **The plan executes its first 10 knots (2 s) per replan and is scored over 30 (6 s).** A floor confined to the first 2 s would
   leave the executed motion identical and change only the scored tail. It is rejected as a design, because it would change the score
   and not the drive.

**Lever chosen for A6:** bound the lift relative to O's own plan, v_f(τ) ≤ v_O(τ) + δ.
* This caps the one-sided overshoot that any level error produces (items 1 and 3; the noise arms of A5).
* It keeps O's own slowdown shape inside the hold windows (item 4).
* It leaves the ramp, the lead cap and κ 1 as in A5.

**Adaptivity, disclosed:** the δ doses are chosen after looking at eval-141 rollouts, and the A6 read is on the same windows.
