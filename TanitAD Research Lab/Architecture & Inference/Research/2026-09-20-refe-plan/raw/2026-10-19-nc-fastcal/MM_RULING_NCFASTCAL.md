# MM ruling on NC-FASTCAL, and a CORRECTION of MM_RULING_PICKRULE §3 (Master Mind, 2026-10-19)

**1. Outcome: NO-RECOVER on both seeds; the C-NULL fired 0 of 5, so the read is not VOID.**
- Calibrated composite vs P2: dNC +0.0029 [+0.0009, +0.0054] / +0.0030 [+0.0009, +0.0055]; dPDMS x100 -1.205 / -1.212.
- The uncalibrated composite reads the same.
- P2's NC head is ALREADY calibrated inside every speed stratum (fitted a 0.99-1.16; held-out BCE gain at most 0.0004 nats).
- ⇒ The NC-calibration line CLOSES, as committed.

**2. ⛔ CORRECTION: `MM_RULING_PICKRULE.md` §3 is WITHDRAWN to HYPOTHESIS** (`RETRACTION_LOG` R53).
- I read Q1 = 0 ("no NC-unsafe EP-led pick has NC sigmoid < 0.5"; median 0.96) as "the NC row is CONFIDENTLY WRONG on exactly the picks an argmax selects, a winner's-curse / Goodhart effect".
- That does not follow. A pick scored 0.96 carries a predicted NC risk of about 4 %. If about 4 % of such picks are unsafe, the row is calibrated, not wrong.
- The NC-FASTCAL agent's reliability-at-picks table (post-hoc, descriptive, no decision authority): observed / predicted NC-unsafe mass at the EP-led picks = **0.895 [0.664, 1.168]**. No overconfidence is visible at the picks. The interval is wide, so calibration is not established either.
- **What stands (MEASURED):**
  - Q1 = 0 and ROW-MISS by the registered rule;
  - no veto threshold recovers;
  - NC / DAC / TTC are FEATURE-LIMITED;
  - DAC has a fast-band deficit;
  - EP concordance is about 98 % slot prior;
  - the EP-led composite loses PDMS (-1.15).
- **What is withdrawn:** "confidently wrong at the selected picks" and "winner's curse".
- **The replacement reading (HYPOTHESIS, not tested):**
  - EP-led picks carry a HIGHER safety risk that the safety rows estimate about correctly.
  - The composite accepts that risk for an EP gain the EP row predicts but does not deliver, because its ranking is mostly the vocabulary's slot order, not scene-specific progress.
  - ⇒ The weak link would be the EP row's REALISED gain, not safety overconfidence.

**3. Consequences for the plan.**
- PRIVGEOM-SAFE stays: sharper safety rows would still let the rule avoid the extra risk.
- The trunk (P6) stays.
- A scene-specific EP read is the other side of the trade. It joins the queue as a HYPOTHESIS to register: "does the EP-led composite's predicted EP gain match its realised EP on S0?" This is a 0-GPU reliability check of the EP row at the picks, on the same footing as the NC table.

**4. Process rule (the agent's question 2: YES).**
- Every future safety- or EP-row registration carries a pre-registered **reliability-at-picks** diagnostic: observed / predicted mass at the composite's picks, per row.
- A threshold share ("score < 0.5") may not be read as "confident error" without it.
