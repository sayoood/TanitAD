# ACTSENS verdict (T0, oracle goal: a world-model diagnostic, never driving performance)

STRATEGIC: PAUSED (PI item 25), not a pass.

| lever | label | next label applied | per chassis |
|---|---|---|---|
| IDM | ACTSENS-NULL | ACTSENS-NULL | {0: 'ACTSENS-NULL', 1: 'ACTSENS-NULL'} |
| DIV | ACTSENS-HARM | ACTSENS-HARM | {0: 'ACTSENS-NARROWS', 1: 'ACTSENS-HARM'} |

## IDM

* acc_mae: BETTER False, WORSE False, beats HOLD at both chassis False; floors c0: 0.0502, c1: 0.0638
* yaw_mae: BETTER False, WORSE False, beats HOLD at both chassis False; floors c0: 0.0101, c1: 0.0159
* MECH-MOVED False; BAND-MET lost []
* next: Lever trained and did nothing. Next box-2 training: a direct action-response objective (DIVRS counterfactual-action target, or a gain-matching term toward IA-3 gain 1.0). Carded next.

## DIV

* acc_mae: BETTER False, WORSE True, beats HOLD at both chassis False; floors c0: 0.0502, c1: 0.0638
* yaw_mae: BETTER False, WORSE False, beats HOLD at both chassis False; floors c0: 0.0101, c1: 0.0159
* MECH-MOVED False; BAND-MET lost []
* next: Lever stays off. IDM next: gamma_L1 300. DIV next: ACTSENS-NULL's next step.
