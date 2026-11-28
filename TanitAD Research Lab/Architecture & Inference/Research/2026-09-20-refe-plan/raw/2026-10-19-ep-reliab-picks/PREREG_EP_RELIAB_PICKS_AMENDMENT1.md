# PREREG EP-RELIAB-PICKS: AMENDMENT 1 (C-POWER is made independent of the real labels)

Written and hashed BEFORE any real-data stage ran and before any real quantity of this arm exists (`PREREG_EP_RELIAB_PICKS.md`, sha256 `e54067d4b16f598c1710064a7f01eafe825d19c01eadbe7124b86fa67e9a8e48`). It governs C-POWER of section 7 and nothing else.

## What was wrong

Section 7 registers C-POWER as: on the real arrays, planting +0.10 in `sigma` at the composite's picks must read OVER, planting -0.10 must read UNDER, and 0 must reproduce the unplanted class. That makes a CONTROL depend on the real baseline: if the real `Delta` or `D_M` is already strongly OVER, a planted -0.10 may fail to cross to UNDER although the instrument is perfectly able to see a 0.10 shift. A gate whose result depends on the number it gates is not a reach check; it is a property of the real data (the same defect `MM_RULING_PICKRULE.md` ruling 1 removed from the landed audit's C-ID3). I found this while writing the tests, before any real-data stage, from the design alone.

## The replacement (supersedes the C-POWER paragraph of section 7)

**C-POWER (planted, label-independent).** For each seed and for EACH form: take the REAL picks (`pk_p`, `pk_e`) and the REAL predicted `sigma` (the composite's EP row), and build a NEUTRAL label field `y*` that equals `sigma` at the two picks of every set (`y*[i, pk_p[i]] = sigma[i, pk_p[i]]`, `y*[i, pk_e[i]] = sigma[i, pk_e[i]]`) and the real EP label elsewhere. On the neutral bank the unplanted `Delta` and `D_M` are EXACTLY 0 (`g = ghat` per set, `M_e = M_p = 1`). Then:
- planting `+0.10` in `sigma` at the composite's pick of every changed set must read `OVER`;
- planting `-0.10` must read `UNDER`;
- planting `0` must read `WITHIN` with an interval exactly `[0, 0]`.
The reads use the real section 6 class rules, the real draws and the real log structure. If any of the six reads (2 seeds x 2 forms x 3 plants = 12) is wrong, the arm is VOID by insufficient resolution. The neutral bank uses no real EP label at a pick, so the check cannot depend on the real baseline. It tests the instrument's resolution of a planted 0.10 shift given the real pick geometry and the real log clustering; it does NOT test the label noise, which is why the half-widths of the real `Delta` and `D_M` intervals are reported beside the result as the descriptive minimal-detectable-effect (no authority).

Everything else in `PREREG_EP_RELIAB_PICKS.md` is unchanged.
