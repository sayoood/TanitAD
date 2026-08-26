# PRE-REGISTRATION — the crossed cell: does freezing the encoder stop drift *without* destroying prediction?

**Registered** 2026-08-26 ~14:50 Europe/Berlin, **before the arm is launched.**
**Author** Master Mind · **Tier** T0-DIAGNOSTIC throughout.

---

## 1. Why this arm, and why it is a *crossed cell*

**E-DEC-61:** trained encoders converge to drift **0.62–0.68** (6 of 7 arms,
7.5k–30k steps, both initialisations); the two frozen arms are the only ones
outside it. The load-bearing evidence is a **within-recipe trajectory** —
`postrain10k` **0.359** → `postrain30k` **0.669**, same recipe, same `--init-from`,
only training length differing.

**Proposed mechanism:** O5 asks that `ẑ_{t+k} ≈ z_{t+k}`. When the encoder is
trainable, the cheapest way to satisfy it is not better prediction but **an easier
target** — a smooth, self-predictable latent. Joint training therefore *manufactures*
drift.

⛔ **C164 is four hours old and says exactly what is missing:** *"a grouped
comparison is a LEVER only if the groups are matched on everything except the label;
the cheapest test is the crossed cell — apply the supposed lever to an arm from the
OTHER group."* ⇒ **This arm is that cell:** the `postrain` recipe, byte-identical,
**plus `--freeze-encoder`.**

---

## 2. ⛔ The failure mode that is already documented, and why drift alone cannot win

`train_v6_staged.py:4690` records **E-DEC-20c**: on a frozen encoder the content
holds to 10k (`n_agents` +0.4035 → +0.4156) **while the predictor goes from ~a
constant to ~5× MISCALIBRATED (`nrmse` 4.83, mean-fraction 0.8174)** and its h=1
head grows 2.676×.

⇒ **Freezing has a KNOWN degeneracy.** A run that lowers drift while wrecking
prediction has not solved collapse — it has **traded one degeneracy for another**,
which is precisely what O13 did (`o13_excess` up, `o5` **+192.4 %** worse). **Drift
is therefore NOT the read. The pair (drift, o5) is.**

---

## 3. The outcomes, committed in advance

Read at step 30,000 against `postrain30k` (drift **0.669**, the matched incumbent)
and its seed replicate (**0.679**, giving run-to-run variance **~1.5 %**).

| outcome | criterion | conclusion |
|---|---|---|
| ⭐ **CONFIRMED** | drift **< 0.45** AND `o5` within **10 %** of `postrain30k` | **Freezing is the lever, with a clean attribution.** The programme's first positive result whose cause is identified. O5's encoder gradient is the drift pump. |
| ⛔ **DEGENERATE** | drift < 0.45 but `o5` **≥ 10 %** worse | E-DEC-20c repeating. **Freezing trades collapse for miscalibration and is NOT a fix.** Report as such; do not retune. |
| ⛔ **REFUTED** | drift **≥ 0.55** | The encoder's trainability is not what sets drift. E-DEC-61's mechanism is wrong, and the 0.62–0.68 attractor needs a different explanation. |
| 🔶 **UNCLASSIFIED** | 0.45 ≤ drift < 0.55 | Between the bands. **State the numbers; do not round them to a verdict.** |

⚠️ **Thresholds are set from measured quantities, not taste.** 0.45 sits below the
trained attractor's floor (0.616) and above the frozen arms (0.199, 0.337); the
10 % `o5` band is the same one O13 was judged against, and is ~7× the measured
seed variance.

⚠️ **The drift read uses `latentmotion.py` unmodified** — the instrument that
produced every number in the table above, including `postrain30k`'s 0.669 on the
same 80 clips. Changing it before the read voids this pre-registration.

---

## 4. Guards

- ⛔ **Report `o5` beside drift in the same table, always.** A drift number alone is
  the O13 mistake.
- ⚠️ **`--freeze-encoder` also freezes the encoder for every other term.** If `o1`
  or `o6` move sharply, say so — the flag is not surgical.
- ⚠️ **One arm, one seed.** Seed variance on drift is ~1.5 %, so a 0.669 → <0.45
  move is far outside it; a *marginal* move is not, and must not be read as one.
- ⚠️ **Every number is T0.** A CONFIRMED result says the transition stops
  manufacturing drift. It says **nothing** about driving.

## 5. Cost, and what it displaces

One 30k arm on Thor (~8 h). **It displaces `postrain30k_seed2`**, killed at ~7k
steps: a third variance point refines an estimate already at 1.5 %, while this cell
decides whether the campaign's leading explanation is right. ⚠️ Stated because
killing a running arm is a real cost, not a free choice.
