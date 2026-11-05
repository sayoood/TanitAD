# PI decision item: the next lever after addendum 4 row K2 (RECOVER-NEXT, 2026-11-05)

Addendum 4 row K2 (RPT read, landed `6ef84e4`, T1-kin) sends the next arm to a PI decision: the per-tick empirical deviation
shape closes 0.0725 / 0.1508 of the M1 crawl excess (R, same law and dose, constant shape: 0.2198 / 0.1318; bar 0.5), its guards
fail at both seeds, and its composite fails D2 on M1 alone (+0.0540 / +0.0553), with D1 and D3 true.

## What six recovery arms measured (T1-kin, MEASURED, `RESULTS.md` of each arm)

| arm | M1 share closed s0 / s1 | ADE6 X − C0 (m) s0 / s1 | guards |
|---|---|---|---|
| R (bank law, p 0.3) | 0.2198 / 0.1318 | +0.1892 / +0.0977 | fail |
| RTL (logged target) | 0.2131 / 0.1933 | +0.1492 / +0.0925 | fail |
| RP01 (bank law, p 0.1) | 0.1473 / 0.1342 | +0.0017 / +0.0008 | fail (non-ADE6 rows) |
| RAPR (a-priori law, p 0.3) | 0.4447 / 0.3175 | +0.3504 / +0.3202 | fail |
| RAPR-P01 (a-priori law, p 0.1) | 0.2147 / 0.1391 | +0.0371 / +0.0429 | fail (along-track, DK) |
| RPT (bank law, p 0.3, per-tick shape) | 0.0725 / 0.1508 | +0.1668 / +0.0576 | fail |

The closure-need estimate is about 0.68 / 0.60 (ESTIMATED, post hoc, `rowk_decision/d2_closure_need.json`). No recovery arm comes within
reach of it inside its guards: closure and harm rise together with the size of the deviation, and the per-tick shape lowered the
closure at s0. Reading (**[H]**, not a rule): the recovery fine-tune family is exhausted at the laws, doses and shapes measured.

## Why the crawl happens, as measured

* The M1 excess enters through the ego-history input: the logged history closes 0.918 to 0.984 of it, the logged speed per tick
  0.599 to 0.633 (oracles, §7.35).
* The frozen visual features carry no motion signal the trunk can use: the model's own four frames add nothing to the go/stay edge
  (§7.31), a finer token grid or an attention read-out does not help (§7.33), and T's five speed cells cannot be read from T's own
  pooled input (SPEEDREAD, NOT IN INPUT, §7.24).
* So O has only its own executed history to set its speed level, and on its own motion that history is the slow one.

## The options

| | (a) level-from-image auxiliary (row K2's committed next) | (c) motion from vision: a perception change |
|---|---|---|
| what | a head regressing the logged speed from the current frame's frozen tokens, fed to O at every tick | let the encoder see motion: encoder adaptation over the frame stack (the LoRA path, AR-2 infrastructure PASS, arm not trained) or a frame-difference input |
| aimed at the measured cause | partly: single-frame frozen tokens give a context prior on speed, not motion (probe R² 0.489 on slow windows, 0.117 on the M1 windows) | yes: it supplies the speed level the history oracle shows O needs, from vision |
| engineering | about 1 to 1.5 days on the dev-box CPU (head, loss, feedback path, flags with gate entries, G-LIVE checks, reader rollout mode) | UNVERIFIED; larger than (a) |
| compute | 2 box-2 runs, at least 1.3 h each pair | UNVERIFIED: training through the encoder likely needs GPU memory and time beyond box 2's cached-feature runs, which is a resource decision |
| P(counted DOMINANT), ESTIMATED | about 0.04 (decision item 2026-11-04) | not estimated |

(b2), the a-priori per-tick ramp named as option (b)'s fallback, is not recommended: RPT shows the per-tick shape lowering closure,
and RAPR's constant-shape closure (0.4447 / 0.3175) is already below the need.

## Recommendation

**(c), the perception change**, because it is the one lever aimed at the measured cause; it needs your architecture decision and,
UNVERIFIED, compute beyond box 2. If you want to stay within the resources allocated now, **(a)** as committed by row K2: it is
engineered on the dev box and trained on box 2, with a low estimated chance. Neither starts without your decision; any box-2 run needs
your go.
