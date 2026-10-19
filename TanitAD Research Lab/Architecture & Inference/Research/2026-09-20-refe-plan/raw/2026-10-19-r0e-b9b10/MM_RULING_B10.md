# MM ruling on R0E-B9B10 escalation 2 (Master Mind, 2026-10-19)

**Question:** PREREG_R0E_PICK §5 is silent on whether R0e-id must have run to its end (not been killed after a clean u250 read) for ID-PASS.

**Ruling: FAIL CLOSED.** ID-PASS requires BOTH:
- (a) the u250 identity verdict reads ID-PASS, by the reader as built;
- (b) R0e-id carries NO `arm_R0e-id_killed` marker, from any kill rule. Under B9, K3 is observe-only for R0e-id, so a kill there is K1 / K2 / divergence / NaN class.

**Why:**
- A kill after a clean u250 is evidence the identity run is not healthy at the horizon R0e-pick trains to.
- R0e-pick is an A40 day; a wrongly admitted launch costs more than a re-run of R0e-id.
- The operator sequence (`RLS1_ARMS="R0e-id"`, then `"R0e-pick"`) already starts R0e-pick only after R0e-id's chain has ended. "Not killed" is therefore the same as "completed" in practice, and costs no wall time.

**Implementation:** apply the one-line fix proposed in `pkg/RESULT.md`, so the `id-gate` refuses when the kill marker exists. Add:
- a planted case: clean u250 verdict + kill marker ⇒ REFUSE;
- a RED mutation: deleting the marker check makes that case pass.
This lands with B9B10, after R0E-B2B5. Record it in RESULT.md as an MM ruling on a prereg silence, not as a prereg change. The registered ID criteria are untouched.
