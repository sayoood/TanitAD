# MM rulings on PICKRULE-AUDIT-NC (Master Mind, 2026-10-19)

**1. Amendment 2 is ACCEPTED, and the read is VALID with its disclosure.**
- The registered C-ID3 reach clause ("a veto at tau = 0.5 changes P2's pick on at least one set") assumed a property of the DATA, not of the code. P2's rule never picks a proposal its NC row puts below 0.6286, so the clause could not pass on any correct implementation.
- It failed at the gate BEFORE any decision quantity existed (`result.json` absent; the gate report is kept). The amendment keeps the failed clause visible as INFORMATIONAL and gates a data-independent reach check instead: a planted, label-free -3.0 logit shift, with a RED mutant.
- ⇒ Class for future preregs: **a reach check must be stated on the CODE PATH (a planted input), never on a property of the real data.** A data property belongs in the results, not in the gate.

**2. The outcome stands: ROW-MISS** (Q1 = 0.000 on both seeds, exact binomial upper bound 0.044; median NC sigmoid at the EP-led NC-unsafe picks 0.960).
- No veto threshold recovers: dPDMS lower CI -1.42 to -1.52 x100 at every tau.
- The EP-led composite here is EP-R1's c* EP row regenerated out of fold, because RL-1 V1's rows do not apply to S0's features (0 of 76 tensors shared). The register says this arm read EP-R1's composite, not V1's.

**3. The joint reading with `D-REFE-FASTNEG-CENSUS-1` is the finding.** The NC row ranks the fast band as well as the slow bands ON AVERAGE (no within-token deficit), yet it is CONFIDENTLY wrong (about 0.96) on exactly the proposals an EP-led argmax selects.
- An argmax over a composite selects for the rows' errors; this is a winner's-curse / Goodhart effect, not an average-resolution defect.
- DAC is worse on both counts: a fast-band deficit, and the larger unsafe-rate rise (+0.0071 vs NC +0.0029, post hoc, no authority).
- ⇒ **No head-only, monotone or threshold lever on the current features can fix a confident error at the selected picks.** Five arms now agree: EP-R1, EP-R1ub, SR-TTC, the census and this audit. The levers that remain change what the rows SEE (features or trunk) or make the selection PESSIMISTIC under row uncertainty.

**4. Next arms, in order:**
- (a) **NC-FASTCAL**, the committed ROW-MISS arm. It is cheap, so it runs, pre-registered. It must be SPEED-CONDITIONAL calibration; a monotone transform of the NC logit cannot reorder confident errors.
- (b) **PRIVGEOM-SAFE** (supersedes PRIVGEOM-DAC; DAC and NC lead / TTC scalars). It localises whether the missing information exists upstream.
- (c) The trunk: P6's end-to-end scorer training, read about 25 Oct.
