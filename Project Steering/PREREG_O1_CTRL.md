# PRE-REGISTRATION — MM-E11: turn ON the objective whose documented job is exactly the defect we measured

**Written** 2026-08-30, **BEFORE the arm runs** · Master Mind · **Tier** T0 for the
primary read; T1 escalation pre-committed below.

```yaml
hypothesis: MM-E11
question: MM-E10 measured the predictor to be ACTION-DEAF (action moves the
          prediction 0.4-0.6% as much as the scene at h=1; ~0.001% beyond).
          Does restoring O1 — the term the code documents as "the ANTI-ACTION-ECHO
          measure", default 1.0, which every arm ran at 0 — make it action-sensitive?
one_variable: --w-o1-ctrl  0.0 -> 1.0   (the CODE'S OWN DEFAULT, not a tuned value)
matched_incumbent: postrain30k   (already measured by MM-E10)
```

## 1. Why this arm and not the frozen-teacher target

The pre-committed next anti-drift lever was the frozen-teacher feature target. ⛔
**MM-E10 says that is aimed at the wrong defect.** A predictor whose action channel
contributes 0.4 % of its scene channel is not primarily suffering from drift —
drift is *downstream* of a predictor that is barely conditioned. That is consistent
with MM-E4 (drift trivially reducible ⇒ a symptom) and MM-E6 (drift self-referential,
in directions the scene does not explain).

⭐ **And the lever is not a new idea — it is a switch we turned off.**
`train_v6_staged.py:6831` sets `--w-o1-ctrl` **default 1.0**; `:215` documents it as
*"response-form L_ctrl, FROM STEP 0"*; `:3116`/`:3143` consume it. **Every arm in
this campaign passed 0.** Verified it is NOT stage-zeroed: the `replace(...,
o1_ctrl=0.0, ...)` calls at `:348`/`:361` are stages **S-T** and **S-S**; our arms
are **S-W**, so the flag will genuinely be in force. *(Checked by source because
`--help` crashes on Thor — MM-C11's lesson: existence is not function.)*

## 2. Arm

`o1ctrl30k` = `postrain30k`'s recorded launch line **verbatim** + `--w-o1-ctrl 1.0`.
Everything else held: stage S-W, 30k steps, batch 8, window 6, lr 1e-4, o5_form l1,
o5_k 8, o1_k 4, horizons [1,2,4], w_o5 1.0, w_o6 0.1, all other O-terms 0, same
init, same parity cache, in_channels 9. ⚠️ `cond_param` stays **None** — postrain30k's
recorded value; using the omega template would silently add a second variable.

**Cost:** ~8.6 h on an idle Thor. Readout is free (the MM-E10 probe re-run).

### ✅ 2.1 — THE ONE-VARIABLE CHECK, RUN 2026-08-30 AT STEP ~18,000, BEFORE ANY READ

§4 committed to *"config-diff must show **only** `w_o1_ctrl` before any read"*. Done —
and it required separating two things a naive diff conflates:

```
postrain30k 194 args · o1ctrl30k 210 args
SHARED KEYS WITH DIFFERENT VALUES:  out (the output directory) · w_o1_ctrl 0.0 -> 1.0
                                    total: 2
KEYS ONLY IN o1ctrl30k:             16   (absent from postrain30k, NOT value changes)
```

⭐ **VERDICT: the one-variable claim HOLDS.** The only genuine difference is
`w_o1_ctrl`; `out` is the output path. The 16 extra keys are flags added to the
trainer *between* the two runs — `w_o13_ego 0.0`, `w_o14 0.0`, `o6_innovation False`,
`ema_decay_ramp 'off'` and so on, all inert.

⛔ **AND THE ONE THAT LOOKED LIKE A SECOND VARIABLE IS NOT ONE — but it took a real
check, not a glance.** `o1ctrl30k` records `cond_param='steer_accel_v'` while
postrain30k records **nothing**. That is exactly the confound §2 warned about by name.
Resolved at source: `COND_INCUMBENT = "steer_accel_v"` (`train_v6_staged.py:164`) **is**
the argparse default (`:6914`), the consumers fall back to it via
`getattr(a, "cond_param", COND_INCUMBENT)` (`:4519`, `:5603`), and the three other 30k
arms all record the same value. ⇒ postrain30k **ran** the incumbent parameterisation and
merely predates the key being *recorded*.

⚠️ **THIS PRE-REGISTRATION WAS ITSELF IMPRECISE AND IS CORRECTED HERE.** §2 said
*"`cond_param` stays **None** — postrain30k's recorded value"*. Its recorded value is
**ABSENT**, not `None`; I read `None` because `dict.get()` returns it for a missing key.
And `None` was never a legal value — argparse restricts to two choices and `_lift3`
raises on anything else (`:3599`). ⇒ the correct statement is **"`cond_param` stays the
INCUMBENT DEFAULT `steer_accel_v`"**.

⭐ The general form, and it is the fourth instance in one day: **an absent key and a key
whose value is null are different facts, and `.get()` makes them look identical.** A
config diff built on `.get()` reports 16 phantom variables here — enough to declare a
clean arm confounded and discard a valid 8.6 h run.

## 3. Reads and outcomes, COMMITTED IN ADVANCE

**Primary:** the MM-E10 action-divergence probe, same corpus, same n, same
instrument. Incumbent values to beat (`postrain30k`): **h1 ratio 0.00595 · h2
0.00002 · h4 0.00002**.

| outcome | criterion | consequence |
|---|---|---|
| **O1-WORKS** | h1 ratio rises by **≥10×** (to ≥0.06) ~~and h2/h4 rise off the floor~~ — see the amendment below | the diagnosis is confirmed and O1 enters the v7 recipe; the frozen-teacher lever is deprioritised |
| **O1-INSUFFICIENT** | ratio rises but stays **<10×** | the term helps and is not the whole story; report the number, do not adopt on its own |
| **O1-INERT** | ratio unchanged within noise | ⛔ the defect is NOT the missing objective — it is architectural (how actions enter the predictor), and the next lever is the conditioning path itself, not another loss weight |
| 🔶 MIXED | h1 and h2/h4 disagree | numbers, no verdict (C160) |

### ⛔ AMENDMENT, MADE AT STEP ~23,000 — BEFORE THE 30k READ, AND HERE IS WHY THAT MATTERS

The `O1-WORKS` criterion as first written was **UNSATISFIABLE**, and would have failed a
succeeding arm. It required *"h2/h4 rise off the floor"* — but those heads are **never
trained**, so they cannot rise. MEASURED on **this arm's own** banked snapshot
`ckpt_step10000.pt`:

```
predictor_op.heads.1.weight   |W| = 6.7488     ← trained
predictor_op.heads.2.weight   |W| = 0.0262     ← at initialisation
predictor_op.heads.4.weight   |W| = 0.0261     ← at initialisation
```

⇒ ~258× smaller, **with `w_o1_ctrl 1.0` in force at step 10,000**. So O1 does *not* train
the multi-horizon heads either, and MM-E14's finding carries onto this arm rather than
being specific to the arms it was found on. An arm showing a **50× h1 gain** would still
have failed the criterion as originally written.

⭐ **AMENDED CRITERION:** `O1-WORKS` = the **h1** ratio rises **≥10×** (to ≥0.06). The
h2/h4 clause is **struck**. ⛔ And h2/h4 must not be *reported* either: they measure
initialisation noise, not conditioning, so quoting them would repeat exactly the error
MM-E14 retracted.

⚠️ **On the legitimacy of amending a pre-registration at all.** Changing a criterion
*after* seeing the data is the cardinal sin and this is not that: the arm is at ~23,000
of 30,000, **no MM-E11 number exists**, and the reason is an instrument defect
independently registered as MM-E14 *before* this arm was launched. The amendment is
recorded here with its evidence and its timing rather than made silently — a
pre-registration that is quietly edited is worth less than none. ⭐ Had this been left,
the failure mode was not a wrong number but a **correct arm discarded**: the most
expensive kind, because it looks like a clean negative.

⛔ **THE ANTI-GATE, COMMITTED BEFORE ANY NUMBER EXISTS.** Banked primary: an arm
**10.7× worse** on open-loop next-action MSE was **2.3× better** closed-loop, because
history helps only once the shortcut is closed. ⇒ **A T0 PREDICTION REGRESSION ON
THIS ARM IS EXPECTED AND IS NOT GROUNDS TO REJECT IT.** If drift rises or nrmse/cos
worsen while the action ratio rises, that is **the intended trade**, and it escalates
to **T1** rather than being failed at T0. *If our own gate rejects this arm on
prediction, the gate is wrong, not the arm.*
⚠️ **The limit, so this is not a blank cheque:** a T0 regression **with no action-ratio
gain** is a dead arm, not a de-confounded one.

## 4. Controls

* The action-divergence probe's own **C0 identity** (same actions twice ⇒ spread
  exactly 0) and **C1 scene reference** must both read their known values, or the
  panel is VOID rather than a result.
* **Incumbent is not re-run** — `postrain30k` is already measured under the identical
  instrument, corpus and n. Config-diff must show **only** `w_o1_ctrl` before any read.

## 5. What this cannot show

Nothing about driving (T0), and nothing about whether action-sensitivity *converts*
into closed-loop skill — that needs T1 with the hold-action control, and is the
escalation this arm earns if it succeeds. ⚠️ Also: a single seed. Deltas smaller than
the (unmeasured at 30k) seed band are **unresolved, not null** — the MM-E8 lesson.
Given the incumbent's h2/h4 sit at 1e-5, a ≥10× criterion is far outside any
plausible seed noise, which is why the primary criterion is set there.
