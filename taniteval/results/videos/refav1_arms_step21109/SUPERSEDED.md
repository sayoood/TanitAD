# ⛔ SUPERSEDED — this reel's FOOTAGE is right and its FRAMING is refuted

**Replacement:** `taniteval/results/videos/refav1_arms_curvature_step21109/`
(rendered 2026-09-06 from the same banked dumps, same checkpoint, same 40
windows / 8 episodes).

## What is wrong with the cut in this directory

Its intro and outro cards call `ccos_argmax` **"the arm that TURNS"** and
present `wk15` as **accurate because it does not turn**, and they draw
**`ha0_ext`** as "the floor". All three readings are refuted by `M74`, `M75`,
`M77` and `M79` in `Project Steering/Decisions/2026-09-06-mm-decisions.md`:

1. ⛔ the turn-recall gate those words rest on — `|dyaw| > 0.15` rad — is
   **UNREACHABLE at the relevant speeds and the HUMAN FAILS IT**: she passes
   **3 of 9** TURN_L windows (median dyaw **0.0431**) while `ccos_argmax`
   passes **6 of 9, twice as often**. *A gate the ground truth fails is not
   measuring skill.*
2. ⭐ on **curvature MAE** — which the human passes by construction — the
   ranking **INVERTS**: `wk15` **0.030982**, `best` **0.031281**, the
   **perfectly straight** floor `ha0` **0.040083**, and `ccos_argmax`
   **0.055369 — worse than a plan that never steers**.
3. ⛔ **`ha0_ext` is the wrong floor for that sentence.** It holds the
   **MEASURED** curvature, swings up to 8.21 m, reads curvature MAE
   **0.077298** and leaves the friction circle on 18.5 % of windows. It is an
   **ADE** floor. The replacement draws **`ha0`**, the perfectly straight plan,
   because *"tracks the road worse than a straight line"* is only checkable
   with the straight line on screen.

## What is still good here

Every **trajectory, decision argmax and world-model curve** in these frames is
read from the same banked dumps the replacement uses, and the **GT integrator
control** (`GT_CONTROL.png`, STEER 0.1552 m vs KAPPA 0.8727 m, 5.62×) applies
unchanged. The **footage** is not in question; the **cards and the white line**
are. Quote numbers from the replacement's `WINDOWS.json`, never from this
directory's cards.
