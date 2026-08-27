# PRE-REGISTRATION — P0: two-term core vs the EMA-teacher target (tiny ladder)

**Written** 2026-08-28 ~04:00 (Europe/Berlin), **before launch** · **Author** Master
Mind · **PI approval** 2026-08-27 ("approve 2 and 3") · **Hypothesis** MM-E1 (the
first Master-Mind-stream experiment id under the prefix rule) · **Tier** T0.

```yaml
hypothesis: MM-E1
question: does replacing O5's LIVE target latents with EMA-teacher latents
          (--o5-target ema; the Drive-JEPA collapse core) change drift,
          prediction quality, or participation at the tiny scale?
one_variable: --o5-target ema           # implemented 8ea7731be, 6 tests,
                                        # resync-after-init-load pinned
matched_incumbent: o14base2k            # the ladder's w=0 arm IS the two-term
                                        # 2k incumbent on this exact line —
                                        # verify by config diff before reading
arms:
  - ema2k_s0: the o14base2k launch line verbatim + --o5-target ema
  - ema2k_s1: same + --seed 1           # the 2k seed-band for THIS recipe
held_constant: [recipe, init distill_init.pt, parity train corpus, steps 2000,
               batch 8, --cond-param omega_accel_v, --tac-vocab-version v6.0,
               w_o14 = 0]
launch: chain pattern (script from a FILE), sequential, first row content-verified
```

## Reads and outcomes, committed in advance

Instruments as the ladder's: `latentmotion` drift, `meanpred` nrmse,
participation. ⚠️ The 2k scale-facts bind: every 2k arm reads at the
mean-predictor floor (nrmse ~0.98) and the pixel-marginal does not exist at 2k —
so the readable quantities are **vs-base relative**, and cos is the most
sensitive of the three (the ladder resolved 0.18 → 0.25 under O14).

| outcome | criterion | consequence |
|---|---|---|
| **EMA-BETTER** | drift lower than base beyond the ema2k seed spread AND nrmse/cos not worse | the P0 bake-off escalates to a 30k matched pair before any v7r adoption |
| **EMA-NEUTRAL** | all three within the seed spread | the two-term core stays; EMA is dropped as redundant complexity (Drive-JEPA's core does not transfer at our scale/recipe) |
| **EMA-WORSE** | drift higher or nrmse/cos worse beyond spread | dropped, with the numbers |
| 🔶 **MIXED** | axes disagree | state the numbers; no verdict (C160) |

⛔ **What this cannot show:** anything at 30k scale (the drift attractor is a 30k
phenomenon), anything about driving (T0), and any absorption read (2k). If
EMA-BETTER, the 30k pair inherits those questions.

**Cost:** 2 arms × ~35 min Thor, after `o14fut30k` vacates. The two-term incumbent
is NOT retrained — `o14base2k` is bit-comparable by construction (verify: config
diff must show only `--o5-target` and seed).
