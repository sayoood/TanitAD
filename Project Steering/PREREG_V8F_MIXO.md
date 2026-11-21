# PREREG_V8F_MIXO: train O on codes mixed PER KIND, so it learns to use a partly right T code (written 2026-10-25, before any instrument, any KIND data or any MIXO data)

**Status: conditional.** It runs only if `PREREG_V8F_KIND` reads DIFFUSE, INTERACTIVE or MIXED. If KIND reads ONE KIND CARRIES, the T-side arm on that kind goes first and this waits. Box 2 needs the PI's go.

**Evidence class of the motivation:** MEASURED (`D-V8F-T-ATTRIB-1`, `raw/2026-10-25-t-attrib/`).
* The T-to-GT ADE6 gap (1.2045 / 1.0508 m) is CONVEX in T's accuracy: the first 25 % of the way to GT buys 6–15 % of the gap.
* 18–23 % of windows get WORSE with the GT constraints.
* Every T-side lever measured moved T's error by 0–6 %.

**The mechanism this targets.** O is teacher-forced on WHOLE codes: one switch per sample, the GT code with probability p_gt (1.0 → 0.2), else T's own (`stack/tanitad/models/v8f_mini.py:620-622` at `93340166`). So O has only ever seen an all-GT or an all-T code, and never a code where some parts are right and some are wrong. That is exactly what a better-but-imperfect T would hand it.

## The arm (ONE lever: the granularity of O's teacher forcing)

* **X = M2b + `--code-mix kind`.** The code O reads is mixed per UNIT, with an independent draw per sample and unit at the same p_gt schedule.
  * The units are the six constraint kinds of `PREREG_V8F_KIND` (speed, angle, tte, timegap, geom, leadgap) plus the decisions (goal + latdec + londec as one unit).
  * Each cell's expected GT fraction equals B's. Only the joint distribution changes.
* Everything else is B's: data, steps (19,164), the S-level switch (`:610-612`, unchanged), T's own losses, the windows and the eval.
* **The default `--code-mix whole` is B**, and the launch gate must prove it bit for bit.

## Proofs before any run (launch gate, `SPEC_REFCV7.md` §2; each with a deliberate-regression arm that must FAIL)

1. **G-DVB:** `--code-mix` has an entry, and the built model's value equals the argv.
2. **Identity:** `whole` reproduces B's built model and a 200-step loss trace exactly.
3. **Analytic identities on the mask:**
   * at p_gt = 1, `kind` equals `whole` (every cell GT);
   * at p_gt = 0, both equal T's own code exactly;
   * at p_gt = 0.5, the per-sample GT fraction must be PARTIAL. Across 4,096 draws its distribution must hold values strictly between 0 and 1. A mask that only ever gives all or nothing is a switch, not a mix.
4. **Reachability from the actual caller:** trace argparse → config → `_forward` → the code O reads, with file:line at each hop. Delete the mixing line and the check must go RED.

## The reading (tier T1-class open loop, eval-141; paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m, 2F = 0.2415 m)

| read | definition | role |
|---|---|---|
| **PRIMARY** | own-code ADE6, X_s − M2b_s, at seed s | **≤ −2F, separated** at seed 0; then at seed 1 (run only if seed 0 passes) |
| **D-M (mechanism)** | the α-dose response r(0.5) of `RESULT_V8F_TAC_A2.md` (C), measured on X with TAC_A2's own definition, against B's 0.36 / 0.21 (M2b s0 / s1, `RESULT_V8F_TAC_A2.md` table, row "(C) α = 0.50") | beside; names the next lever on a FAIL |
| **HARM** | the GT-code oracle ADE6 on X against B | beside (O now sees fewer all-GT codes); not deployable, never a bar |
| families | longitudinal (incl. distance keeping K = 10), lateral, tactical, strategic (NOT PAUSED for v8F-mini), Λ | beside, per family, never pooled |

| outcome | condition | next |
|---|---|---|
| **PASS** | PRIMARY met at both seeds | X becomes the base; KIND is re-read on X |
| **FAIL, O LEARNED** | PRIMARY missed at seed 0, r(0.5) rises by ≥ 0.10 over B's | O now consumes partial codes, but T is too wrong where O reads it: the T-side arm on the kind KIND ranks first |
| **FAIL, O DID NOT LEARN** | PRIMARY missed at seed 0, r(0.5) rises by < 0.10 | O cannot use a mixed code at this capacity or with these inputs: an O-capacity or O-input arm, under its own PREREG (an architecture change: PI go) |

Seed 1 runs only after a seed-0 PASS. The rig's run-to-run floor is the seed-paired swing (0.102 m ADE6, R66). An effect below it is not reported as a lever effect, whatever its interval.

## Refusals (rc 2)

* No launch-gate PASS token for the exact commit, argv and data manifest.
* Any identity or analytic check above misses.
* The eval's windows, episodes or checkpoint sha256 differ from the banked run.
* A family is missing.
