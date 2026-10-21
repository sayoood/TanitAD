# PREREG R8-PICKER, ADDENDUM 1 (K6 scope)

* **Clock read before sealing:** 2026-10-21 21:03:47 +0200 (Berlin, dev-box clock).
* **Parent:** PREREG_R8_PICKER.md, sha256 e6ec30e7bc03b3dc4811674352f94c9f128203cbf814f284876f239ae867b258.
* **When sealed:** BEFORE any capture roll, including the CPU smoke. `runs/` and `smoke/` hold no roll output.

## Why

K6's first half reads "on windows with `v_max_valid == 0`, `ceil_keep` is all True". That half is VACUOUS on eval139.
MEASURED on the banked tree `step48852ema_pf/dump_s1/refcv6_extras.npz`: **0 of 4,754** windows carry
`v_max_valid == 0`. The fed ceilings are {8.333, 13.889, 27.778, 33.333} m/s.

## Change

A control that cannot fail checks nothing. K6 is therefore read as follows, and nothing else in the parent changes.

* **K6a (kept).** The vacuous half is reported with its n (expected 0). It is not counted as evidence.
* **K6b (NEW, blocking). All-pass where the cap is not binding.** On every window where EVERY one of the 149
  candidates has `refcv6_selection.planned_max_speed` (the model's own function, horizons 5,10,15,20,30,40,50,60
  ticks, tick 0.1 s) at or below the fed ceiling from `clamp_posthoc_r8.fed_ceiling` (the eval's ceiling
  derivation), the captured `ceil_keep` must be all True.
  * Expected: True on 100 % of those windows. The count n is reported.
  * The same computation also reports the agreement of `ceil_keep` with (pm <= ceiling, empty row -> all True) on
    every window. That agreement is informative only, because it re-derives the producer's rule.
* **K6c (kept from the parent).** The SHIP pick lies in `e9_keep` on 100 % of windows.

## Smoke windows (CPU, K1)

2 episodes × 4 windows, chosen so that K6b has non-binding AND binding windows. They are listed in
`smoke/windows_smoke.json`:

| clip_index | clip id | windows | non-binding by the slot proxy on the fanceil s1 fan | binding |
|---|---|---|---|---|
| 8 | clip-sha12-ea6d261f6c62 | 9, 14, 19, 104 | 9, 19 | 14, 104 |
| 16 | clip-sha12-c45d6a86ff06 | 11, 16, 31, 76 | 11, 16 | 31, 76 |


> **Banked copy, redacted (Master Mind 2026-10-21):** two PhysicalAI clip-id prefixes in the table above are replaced by sha12 labels (clip-sha12-<sha256(clip_id)[:12]>). The sealed original stays on the dev box (`C:/Users/Admin/refcv8_pf/picker/PREREG_R8_PICKER_ADDENDUM1.md`), sha256 `994bd84910dd920deec948bdc2217b6948e8729121cf0eea3becf8329de1ba89`, which matches its seal file; this copy's own hash differs by construction.
