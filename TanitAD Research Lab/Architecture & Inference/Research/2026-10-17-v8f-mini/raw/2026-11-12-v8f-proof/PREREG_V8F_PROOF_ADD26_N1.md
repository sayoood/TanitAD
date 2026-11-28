# PREREG_V8F_PROOF_ADD26_N1: note to card ADD26 (DISTRULE), frozen before any number of the card is computed on an arm of the card

Written 2026-11-15 after the build of the rule tool (`distrule/DISTRULE_DEFS.md` E3b and E5, `distrule/HANDOFF.md`) and before any rollout of an arm of card ADD26. The card file is unchanged. Two findings of the build change how the rule and one control are run; both were measured on unit cases and on a 16-window CPU smoke of P_s0, an arm that is not in the card, with no cell of the card computed.

## F1. The projection as written advances plans

The card writes a'(i) = max(a_f(i), min(a_w(i), a_c(i))) with the braking floor a_f outside the minimum. Wherever the plan of the model itself slows faster than 6.0 m/s2, or starts more than 1.2 m/s below the measured speed, a_f is above a_w and the new plan is ahead of the plan the model issued, with or without a cap from the lead. On the smoke the rule changed 8 ticks in 5 of 16 windows by this lift alone. This contradicts the first sentence of the card (the plan is slowed) and section 2 (the floor is the braking limit of the rule).

**Amendment A1.** The projection run on every arm of the card is a'(i) = min(a_w(i), max(a_f(i), a_c(i))) (tool flag `--floor cap-only`): the rule never moves a plan ahead of the plan the model issued, and the braking limit bounds only the braking the rule adds. Everything else of section 2 is unchanged (constants 3.0 m, 1.2 s, 6.0 m/s2; saturated step = a step where the floor decides against the cap). Every name of the card is read on this form. The form as written is not run on the arms of the card.

## F2. The constants of control r2 do bind

With TAU = 0 the term max(D, TAU * v) is 0 whatever D is, so the cap of control r2 is the position of the lead itself and binds in windows where the rollout reaches the lead.

**Amendment A2.** Control r2 is run in the mode `plumb` of the tool: the whole path of the wrapper is active (source, gap of the rolled ego, lead speed, logging) and the cap and the floor are absent; its dump must equal rule off through the wrapper, array for array. Added control r2b, from the rule log of every arm of the card: the new plan is never ahead of the issued plan (max over ticks and steps of a' minus the arc length of the issued plan at most 1e-9 m).

## Unchanged

Arms, sources, reads, outcomes, the other controls, the committed next steps, cost and limits of card ADD26. Comparison of dumps for r1 and r2 is by arrays (the dump files carry archive time stamps, so file hashes differ between equal dumps).
