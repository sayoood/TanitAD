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

---

## OUTCOME — read 2026-08-28 ~10:30 (both arms chain-run overnight, config-diff CLEAN)

**🔶 MIXED — the axes disagree; per the committed table: the numbers, no verdict (C160).**
All MEASURED (dev-box 4060, T0-DIAGNOSTIC), raws `p0_drift.json` / `p0_nrmse.json`:

| read | o14base2k (two-term) | ema2k_s0 | ema2k_s1 | seed spread |
|---|---|---|---|---|
| drift r | 0.4531 (t 45.18) | **0.4002** (t 34.20) | **0.3644** (t 29.93) | 0.0358 |
| meanpred nrmse | 0.9876 | 0.9912 | 0.9954 | 0.0042 |
| cos (centred) | 0.1845 | **0.1336** | **0.0979** | 0.0357 |

- **Drift: BETTER on both seeds, beyond the seed spread** (smallest gap to base 0.0529 >
  spread 0.0358; −11.7 % / −19.6 % rel). ⭐ This is the **first measured lever that moves
  drift downward on the trainable line at any scale** (A7/k4 and O14 both left it flat).
- **Prediction: WORSE on both seeds, beyond the seed spread on cos** (−27.6 % / −46.9 %
  rel; base−s0 gap 0.0509 > spread 0.0357). nrmse worse by +0.4 %/+0.8 % (small; every 2k
  arm sits at the mean-predictor floor per the scale-facts).
- EMA-BETTER fails on "nrmse/cos not worse"; EMA-WORSE fails on "drift higher". ⇒ MIXED.
- **Participation was NOT read** — neither probe emits it; recorded as an instrument gap
  (a missing read is a work item, not a silent omission). At 2k with a MIXED no-verdict
  it does not change the outcome class.

**HYPOTHESIS (unread, stated for the record):** an EMA teacher lags the student most at
the START of training — at 2k the target is dominated by near-random early weights, so
the cos cost may be a warmup transient rather than a real trade. Discriminating: a 30k
matched pair, where the teacher has converged. **Not auto-escalated** (only EMA-BETTER
escalates by the committed table); the 30k-pair question goes to the PI as a
decision-with-default in the midday report (default: run it after the v7r launch decision,
Thor is idle).

---

## STAGE 2 — the 30k pair (PI-APPROVED 2026-08-29: "approve the ema 30k pair, run it now")

```yaml
hypothesis: MM-E1 (stage 2)
arm: emao14_30k = the o14fut30k launch line VERBATIM + --o5-target ema
matched_incumbent: o14fut30k (exists; drift 0.6709, nrmse 0.8288, marginal ABSORBED)
one_variable: --o5-target ema          # vs o14fut30k; config-diff verified before any read
why_this_incumbent: the decision is "does EMA join the v7r RECIPE", and the recipe
  now carries O14 — so the recipe-relevant pair is against the O14 arm, not the
  bare two-term postrain30k (kept as context row only). NOTE cond_param stays
  steer_accel_v (o14fut30k's recorded value; the omega template would have
  silently added a second variable).
interpretation_limit (committed): the 2k evidence was collected WITHOUT O14
  (w_o14=0 arms). If this arm reads null/confusing, "EMA fails at 30k" and
  "EMA x O14 interaction" are NOT separable from this pair alone (the A7
  lesson); the diagnostic arm (ema, no o14) would be a follow-up, not a rerun.
reads: latentmotion drift vs 0.6709 · meanpred nrmse vs 0.8288 · the slim
  absorption marginal (must STAY inside null — absorption must not resurface) ·
  cos_ctr vs 0.6043
outcomes:
  EMA-JOINS:   drift meaningfully below 0.6709 (beyond the ~1.5% band) AND
               nrmse/cos within +10%/band AND marginal stays absorbed
               => --o5-target ema enters V7_RECIPE §5.1
  EMA-OUT:     drift ~unchanged OR prediction pays beyond band => dropped; the
               drift attractor stays open (frozen-teacher feature target is the
               next pre-committed lever)
  MIXED:       axes disagree => numbers, no verdict, PI decides with the scaled
               run's timeline in view
cost: 1 arm x ~8.2 h Thor (idle); EMA trainer already installed + content-verified
launch: chain_ema30k.sh, ZZLAUNCHED 07:27 UTC 2026-08-29, first row content-verified
