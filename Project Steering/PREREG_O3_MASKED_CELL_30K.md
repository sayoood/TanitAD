# PRE-REGISTRATION — `o3p30k`: does masked-cell prediction put scene content into the latent?

**Registered** 2026-08-25 18:20, at step ~12,800 of 30,000 — **before the outcome
is known**, and with the abort criterion fixed so it cannot be tuned afterwards.
**Tier** T0-DIAGNOSTIC · **Author** Master Mind

---

## 1. Why this arm exists

E-DEC-40 measured that **Δz is the latent's own drift** (64 % predictable from
`z_t`, t 65.6) and that the drift-removed residual is **noise**. The live recipe
is **O5 + O6 only** — `--w-o1 0 --w-o2 0 --w-o3 0` — and E-DEC-40 showed **O5 is
satisfiable by drift**. O6 is anti-collapse only.

⇒ **Nothing in the objective rewards encoding scene content.** O3 (masked-cell
prediction) is the one available term that directly does: mask cells, predict
them, forcing spatial structure into the latent.

⭐ **And it has never been fairly tested.** `rdw8o3` and `rdw8o23` ran at **2,000
steps only**. Gate B established that arms separate at **30k**, and E-DEC-21
showed the sub-30k regime is **uninformative**. "O3 does not help" was measured in
exactly the regime we know cannot tell us.

**Arm:** `ok8p30k`'s flags verbatim, plus `--w-o3 1.0 --o3-mode static`. One flag
changed. Parity preserved (`physicalai-train-e438721ae894`, seed 0, 30k, batch 8).

---

## 2. ⛔ THE ABORT CRITERION, FIXED BEFORE THE ANSWER IS KNOWN

**Observed at registration (steps 200 → 12,800):** `o3_loss` oscillates around
~0.35 with **no downward trend**, `o3_visible_err` flat at ~0.40, and **`o5_loss`
has risen ~7×** (0.0490 → 0.3127) against the O5+O6 baseline's median **0.134**.
That is the profile of a term adding **gradient interference without
information** — the same shape argued for PSG, which supervised occupancy the
latent already carried.

> **ABORT IF:** `o3_loss` at step **20,000** is **not below** its value at step
> **5,000**. The term is then inert, the arm stops, and ~5 GPU-hours are freed.
>
> **CONTINUE IF:** it is falling — the arm runs to 30,000 and is scored properly.

⚠️ **Why a criterion and not a judgement call.** Aborting now, at 43 %, would
repeat the programme's own documented error: judging an arm in the regime
E-DEC-21 showed is uninformative. Deciding *after* seeing step 20,000 would let me
tune the threshold to the outcome. Fixing it here is the only version that means
anything.

---

## 3. Outcomes, committed in advance

| outcome | reading | what follows |
|---|---|---|
| ⭐ **CONFIRMED** | held-out `n_agents` / `occ_center` / `n_free_cols` **above `ok8p30k`** AND `o5` within +10 % | The objective CAN put content in the latent. O3 enters the recipe and the "initialisation is the only lever" reading is wrong. |
| ⚠️ **PARTIAL** | content up, `o5` degraded >10 % | A trade, not a win. Sweep `--w-o3` down before spending more GPU. |
| ⛔ **REFUTED** | content not above baseline | **The ninth objective term to fail** (O1, O2, O3, O7, O8, O9, O10, O11 precede it). That materially strengthens the case that the lever is **INITIALISATION**, not the loss — the one arm carrying content (`splitp30k`) got there by *starting* from a distilled encoder. |
| ⛔ **INERT** | abort criterion above trips | Not a verdict on O3's *idea*, only on this configuration. Record and move on. |

---

## 4. What would make this test invalid

1. ⚠️ **`--o3-mode static` vs `action`.** This arm uses `static`. A null here does
   not test the `action` mode, and the distinction must be stated in any claim.
2. ⚠️ **`o5` degradation confounds the content read.** If `o5` ends far above
   baseline, a *lower* content number could be a capacity trade rather than O3
   failing. Report both, never the content alone.
3. ⚠️ **The scoring panel must be the SAME instrument** — `spatialenv.py`,
   held-out, lead-matched (`SPD_MIN_LEAD=20`), on identical rows to
   `ok8p30k`/`splitp30k`. A forked copy would drift (C154 #4).
