# AMENDMENT 1 to the P6 read code pin (2026-11-08 ~18:20Z, Master Mind)

**Written before any P6 value was read.** P6 training has not ended and no P6 checkpoint has been pulled.

`selc6_analyze.PINS` accepts two further values:
- `selc CODE_PIN.json`: `255c75e117803d42b4389aa3ac04da2a5ea74f82965dc40681bcb97e43498329`
- `selc_dump.py`: `a742ec952a5493107deb8b228350be5ffe77e966b45b0190ea91838d10f482d1`

**Why.** The confirmation run's `consolidate` had a defect in row order (`refe_selection_confirm_20261107/AMENDMENT_2_CONSOLIDATE_ORDER.md`). The fix is one sort inside `selc_dump.consolidate`, and it re-pins that package. The P6 read imports `selc_dump` but has its own consolidate (`selc6_dump.consolidate`), so no P6 behaviour changes. `CODE_PIN.json` of this package is re-pinned with `make_pin.py`, and the u400 control is re-run.
