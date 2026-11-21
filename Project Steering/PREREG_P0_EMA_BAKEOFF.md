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

### STAGE 2 AMENDMENT — the τ-schedule interpretation frame (recorded BEFORE the read, 2026-08-29 ~14:15)

LAB-RUN-003's Arch package (`…/Research/2026-08-29-ema-warmup-rollout-composition/`)
lands three scope facts that bind the read:
1. **Our arms run FIXED τ=0.996, no ramp** (MEASURED, `train_v6_staged.py:6401`).
   Every published EMA recipe ramps τ because a fixed-τ teacher is dominated by
   random weights early (Mean Teacher / data2vec / BYOL's 2.7–4.1-pt constant-τ
   ablation). At 2k the teacher horizon was 12.5 % of the run — 10–100× off every
   published operating point; **at 30k fixed τ=0.996 sits INSIDE the published
   band** ⇒ a 30k null/negative is a REAL EMA result, not a schedule artifact.
2. **Prediction committed now**: the transient account predicts the cos gap
   NARROWS at 30k. **Symmetric caveat, equally committed**: the 2k drift gain
   may share the transient origin — if drift reduction ALSO vanishes at 30k,
   that is EMA-OUT, not "mixed again".
3. **No published EMA study exists at ≤5k steps** — the 2k stage-1 verdict is
   off the published map in BOTH directions and is hereby down-weighted to
   motivation-only status.
Consequence edits to the outcome table (narrowing, not weakening): EMA-JOINS
additionally requires a **τ-ramp arm before recipe adoption** (published
best practice; adoption on a fixed-τ arm alone would ship a known-suboptimal
schedule); EMA-OUT's follow-up lever is the frozen-teacher feature target
(SALT ≥ V-JEPA 2 at ⅓ FLOPs — banked).

---

## STAGE 2 OUTCOME — read 2026-08-29 ~18:10 (config-diff CLEAN: {o5_target ABSENT→ema, out} only)

All MEASURED (dev-box 4060, T0-DIAGNOSTIC, rig valid; ckpt md5 `3a030ba2` both sides):

| read | emao14_30k (EMA) | o14fut30k (incumbent) | committed criterion |
|---|---|---|---|
| drift r | **0.6952** (t 154.6) | 0.6709 | below beyond ~1.5 % band → **FAILS: +3.6 % ABOVE** |
| meanpred nrmse | **0.7466** | 0.8288 | within +10 % → **−9.9 % BETTER** |
| cos (centred) | **0.7524** | 0.6043 | within band → **+24.5 % BETTER** |
| absorption marginal | +0.0028 (t 2.16) INSIDE_NULL | −0.0047 | stays absorbed → **HOLDS** |

**Verdict per the τ-amendment: EMA-OUT on the committed question.** The amendment's
symmetric caveat fires exactly: the 2k drift gain (−12/−20 %) VANISHED AND INVERTED at
30k (+3.6 %) — it was a warmup transient, as was the 2k cos cost (−27/−47 % → **+24.5 %
gain**). Both stage-1 effects were artifacts of a fixed-τ teacher read far outside its
operating band; the committed prediction ("the cos gap narrows at 30k") is confirmed in
the strongest form. **EMA is NOT the drift lever. The drift attractor stays open**; the
next pre-committed lever is the frozen-teacher feature target (SALT-class, banked).

⭐ **NEW FINDING, exploratory (not pre-registered — registered as E-DEC-69 for its own
follow-up): the EMA teacher is the LARGEST PREDICTION improvement ever measured on the
trainable line** — cos 0.6043 → 0.7524, nrmse 0.8288 → 0.7466, with absorption preserved,
at a +3.6 % drift cost. Whether v7f SHIPS with EMA for prediction quality is a **PI
decision** (the MIXED consequence clause: PI decides with the scaled-run timeline in
view). If adopted, the amendment's τ-ramp-before-adoption requirement applies unchanged.
⚠️ The committed interpretation limit also stands: this pair cannot separate "EMA at 30k"
from "EMA×O14 interaction" (the 2k evidence was collected without O14).

---

## STAGE 3 OUTCOME — the τ-RAMP arm, read 2026-08-30 ~08:40 (Europe/Berlin)

**The amendment's gate: "EMA-JOINS additionally requires a τ-ramp arm before recipe
adoption (adoption on a fixed-τ arm alone would ship a known-suboptimal schedule)."
This closes it.** Arm `emao14_30k_tauramp` = the `emao14_30k` line verbatim +
`--ema-decay-ramp cosine`. One variable. 30k steps, finished 06:1x UTC on Thor.

All MEASURED (dev-box RTX 4060, **T0-DIAGNOSTIC**), raws `tau_drift.json` /
`tau_nrmse.json` / `tau_absorb.json`; ckpt md5 `a64aa48a` verified both sides.

| read | τ-RAMP (cosine) | FIXED τ=0.996 | Δ | rel |
|---|---|---|---|---|
| drift r | **0.6936** (t 148.16) | 0.6952 (t 154.57) | −0.0016 | −0.23 % |
| meanpred nrmse | **0.7408** | 0.7466 | −0.0058 | −0.78 % |
| cos (centred) | **0.7513** | 0.7524 | −0.0011 | −0.15 % |
| absorption marginal | −0.0019 (t −1.49) **INSIDE_NULL** | +0.0028 (t 2.16) INSIDE_NULL | — | holds |

**⭐ VERDICT: τ-RAMP IS NEUTRAL. The recipe keeps EMA at FIXED τ=0.996 and the
cosine ramp is DROPPED as unnecessary complexity.**

Every difference is **~20× smaller than the only seed spread this recipe has ever
produced** (0.0358 on drift / 0.0357 on cos, the 2k stage-1 band). The ramp neither
helps nor hurts on any of the three axes, and absorption stays inside the null both
ways.

⚠️ **Single seed at 30k** — there is no 30k seed band, so the honest statement is
*"indistinguishable at the resolution we have"*, not *"identical"*. The 20× margin
against the 2k spread is what makes the neutral call safe rather than merely
unrefuted.

⭐ **WHY RUNNING IT WAS STILL RIGHT.** The amendment's reasoning was that a fixed-τ
teacher is dominated by random weights early and every published recipe ramps for
that reason. That argument is sound and could not be dismissed a priori — but it
turns out **not to bind at 30k**, because (per the Stage-2 amendment's own scope
fact) fixed τ=0.996 at 30k already sits INSIDE the published operating band. The
ramp was the cheapest way to convert "known-suboptimal in general" into "measured
irrelevant here", and it retires the last open flag in the v7 recipe.

⭐⭐ **A SECOND RESULT FELL OUT FOR FREE, AND IT IS THE MORE USEFUL ONE — A NOISE
CONTROL FOR T1's S-RATE.** The τ-ramp and fixed-τ arms are **T0-indistinguishable**
(above). At T1 (D-T1-V7-READ, same 40 episodes, same control) their S-rates land on
**OPPOSITE SIDES of the same hold-action control**: ramp cl **0.1404 < ha 0.2105**,
fixed cl **0.2807 > ha 0.2105**. ⇒ **Two models that cannot be told apart at T0 give
opposite S-rate verdicts at T1**, which is direct evidence that the S-rate gap is
NOISE at n=40 episodes and must not be read as signal. This retroactively justifies
declining to draw a verdict from the S-rate/distance-metric disagreement (C160), and
it means the **distance metrics — unanimous across all three arms — are the ones to
trust**. An unplanned arm supplied the control the panel lacked.
