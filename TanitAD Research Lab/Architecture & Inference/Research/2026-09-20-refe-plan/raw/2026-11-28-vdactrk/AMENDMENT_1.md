# PREREG_VDAC_TRK AMENDMENT 1: interpretation of five gaps (MM, 2026-11-28)

**Status:** a dated amendment to the FROZEN card `PREREG_VDAC_TRK.md` (sha256 `099a429f...`). It is frozen by sha256 in `AMENDMENT_1.sha256`.
- Made **before any head, dump, calibration or navtest number exists.** None can exist: no head has been trained.
- **It fixes INTERPRETATION only.** No bar, grid, log set, arm or next arm changes. Each item records how the card's own text is read where it was silent. The reader `vdt_read.py` (build B5) implements exactly these.

1. **Only HT-s1 has no positive HC40 margin.** Then VT-r is not read, R fails, and the outcome is **SINGLE-SEED** if P1, P2, M and G hold. CAL-NULL applies only when HT-s0 (the primary) has no positive margin.
2. **The family guard binds on VT, the primary.** VT-r's four families are reported, and its replicate role is carried by R.
3. **A control or confirmation input that is absent reads PENDING, never PASS.** If P1, P2, M, G and R hold while CTRL-T or the families are still PENDING, the outcome is **ADOPT-CANDIDATE**, the card's "PROVISIONAL" state. It becomes ADOPT only through `confirm`, with CTRL-T PASS and all four families present.
4. **P1 holding while M fails matches no committed outcome.** NULL needs P1 to fail, MECHANISM-SMALL needs M to hold, and ADOPT needs M. Such a result is reported as **Unnamed**, as section 7 requires: a PDMS gain without the drivable-area mechanism is not this lever's claim.
5. **"Positive gain" in the calibration means > 1e-9 PDMS** (× 100 units). Ties in the margin grid go to the smaller m, as the card says.
