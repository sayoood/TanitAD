# Card ADD109 (LOOKSWITCH-T0): the fixed-window fit while the head's direction is inside the fixed narrow window, the fit at the head's direction otherwise

Pre-registration of the v8proof fork, written 2026-11-22 after row 174 (card ADD108) was read and before the tool of this card is written. Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance). No encode, no fit: a CPU read of dumps and direction files that exist.

## 1. Question, and what was known when it was written

Row 174: the gap head with both windows at the head's direction reads the gap better where the lead is off the camera axis and worse where it is near the axis, and overall the difference from the fixed windows is not separated. This card asks whether a mix chosen by the head's own direction keeps the first and drops the second.

Known before this card was written, and so a limit of whatever it finds: the bins of row 174 by the azimuth of the lead in the label, for fit seed 0, were open, and they are the reason this rule was formed. The evaluation clips are the same 141. The rule's one number, 8 degrees, is the half width of the fixed narrow window of card ADD99 and is not fitted. A name of this card is a T0 statement about these 141 clips; it would need clips not yet read to be more than that.

## 2. The mix

For fit seed k of 0, 1, 2 and each frame: the gap of S_k is the gap of D_k (the 500-clip fit on the fixed windows, fit seed k) when the head's direction of the frame is inside the fixed narrow window, that is, absolute azimuth at most 8 degrees and absolute difference of the elevation from -2 degrees at most 8 degrees; otherwise it is the gap of P_k (the fit of card ADD108 at the head's direction, fit seed k). The head's direction is the direction file of the frame of card ADD108 (locator fit of seed 0 of card ADD107). No label enters the choice; presence at inference comes from the reader as before.

## 3. Rows, measure, names

Row set: the 3,841 rows of row 174. E(x): mean absolute error of the gap of x against the label gap. R_S = mean(E(S0), E(S1), E(S2)) / mean(E(D0), E(D1), E(D2)); interval: cluster bootstrap over the 141 evaluation clips as in card ADD108 (per-clip sums, mean over the three fits in numerator and denominator), 2,000 draws, `numpy.random.RandomState(109)`, two-sided 95 % percentile; fits held fixed.

* `LOOKSWITCH-T0-CUTS`: the upper end of the interval of R_S is below 1, and each of E(S0), E(S1), E(S2) is below the smallest error of the five 500-clip fits on the fixed windows (3.913 m on record).
* `LOOKSWITCH-T0-CUTS-NOT-EACH-FIT`: the upper end is below 1 and at least one of the three is not below that error.
* `LOOKSWITCH-T0-WORSE`: the lower end is above 1.
* `LOOKSWITCH-T0-NOT-SEPARATED`: every other case with an interval. With no interval the reader stops with no name.

Printed with no name: the count of rows with the direction outside the fixed narrow window, per clip count of clips that have such rows; E of S_k, D_k, P_k on the inside rows and on the outside rows; the same ratio and interval for the inverse mix (P_k inside, D_k outside) against D; the mix with the bound of 4 degrees and of 16 degrees in place of 8 (two values each, ratio and interval), as a view of how much the one number matters; E of S_0 by bin of the label gap and of the absolute azimuth of the label's lead, as in row 174.

## 4. Controls (all have to pass before any name or value is opened)

* w0 (the inputs are those of row 174): on the 3,841 rows the reader finds E(D0) 3.960, E(D1) 3.913, E(D2) 4.112, E(P0) 3.895, E(P1) 3.699, E(P2) 3.689 m within 1e-3 m, and from the direction files S4 0.8748 and `S8_out` 0.5844 within 1e-4.
* w1 (the mix is what the card says): by a second routine written apart, on every row the gap of S_k equals the gap of D_k where the direction is inside and the gap of P_k where it is outside (largest absolute difference 0); the inside and outside rows together are the 3,841 rows.
* w2: the selftest of the tool passes on the pod before the read (a synthetic world in which a planted error of +5 m on the outside rows of P_k raises E(S_k) by exactly 5 m times the share of outside rows, and a planted swap of inside and outside is caught by w1).

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card. None of w0 to w2 depends on the subject.

## 5. Families

T0, no rollout and no plan: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC are not read (n = 0 for each, with this reason); the measure is a T0 read of one input of the lead rule, which acts on LONGITUDINAL (distance keeping) once it feeds the rule.

## 6. Cost, what follows

Minutes of CPU on the fork pod; stop rule 15 minutes. `LOOKSWITCH-T0-CUTS`: the mix is carried as a possible gap input for a later rollout card, and only if card ADD110 says the gap carries the price in falling behind; nothing is adopted. Every other name: reported as written; the line of windows at the head's direction then rests on the 500-clip pilot as measured in row 174. Adoption of any input into the candidate spec is not part of this card.
