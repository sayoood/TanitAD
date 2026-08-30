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

## 3. Reads and outcomes, COMMITTED IN ADVANCE

**Primary:** the MM-E10 action-divergence probe, same corpus, same n, same
instrument. Incumbent values to beat (`postrain30k`): **h1 ratio 0.00595 · h2
0.00002 · h4 0.00002**.

| outcome | criterion | consequence |
|---|---|---|
| **O1-WORKS** | h1 ratio rises by **≥10×** (to ≥0.06) and h2/h4 rise off the floor | the diagnosis is confirmed and O1 enters the v7 recipe; the frozen-teacher lever is deprioritised |
| **O1-INSUFFICIENT** | ratio rises but stays **<10×** | the term helps and is not the whole story; report the number, do not adopt on its own |
| **O1-INERT** | ratio unchanged within noise | ⛔ the defect is NOT the missing objective — it is architectural (how actions enter the predictor), and the next lever is the conditioning path itself, not another loss weight |
| 🔶 MIXED | h1 and h2/h4 disagree | numbers, no verdict (C160) |

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
