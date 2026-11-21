# SPEC_R1 amendment A1 — the dense-label COVERAGE control is restated on the population the bars score

*Registered by the Master Mind 2026-10-04, BEFORE any R1 arm (H1–H5c) was read against any §6 bar. What had been read:
only the coverage control's own value on the scored eval windows (R1's report to the Master Mind).*

**The registered control (SPEC_R1 §6):** "dense-label coverage ≥ 0.95 of scored windows".

**Why it is mis-specified (MEASURED, R1 cache, 4,634 scored eval windows):** coverage reads 0.8677, which equals exactly
the share of scored windows whose 6-s future lies inside the clip (slot-60 valid = 0.8677); there are 0 IGNORE windows
with a valid slot 60. No label that needs the ego's next 6 s can exist on a window ending within 6 s of its clip end —
the control measured clip truncation, not label coverage. The route rule leaves exactly those windows "unclassified";
they enter no route bar (§6 bars read GT-turn and GT-straight windows only).

**A1 (replaces that one control; nothing else in SPEC_R1 changes):** the dense-label coverage control PASSES iff
coverage ≥ 0.95 on EACH classified class (turn-left, turn-right, straight, gentle) of the scored windows; the
unclassified share and its coverage are reported beside it. The label-validity control (dense lateral side vs the GT
turn class ≥ 0.95 on GT-turn windows), H2s's agreement control and every other registered control and bar are unchanged.
Measured at registration: 1.000 on every classified class; validity 0.9991 on 2,317 GT-turn windows.

**Lesson carried to SPEC_REFCV8:** a coverage bar on a future-window label is stated on the windows that HAVE that
future (or per scored class), never on all windows.
