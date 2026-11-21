# AMENDMENT 7 result: NOT CONFIRMED (2026-11-08)

Committed test, on `090d6a17e77d59b5`, 10 Hz box-centre path, 171 ticks, the same arrays the selector and scorer build (MEASURED, `2026-11-08-suite-selection/a7/MEASURE_090d6a17.json`, stream A7-IMPL, not re-run by the MM):

- First v < 0.5 tick: 77, with 31.701 m of arc up to it.
- A = 0.203 m, N = 0.052 m, so A − N = 0.151 m against a bar of 0.5 m.

**Rulings 1-5 are not applied. No code, pin or candidate list changes.** Standstill jitter adds about 0.15 m on this path, not 3.76 m.

**Where the 3.76 m is, so far (HYPOTHESIS):** the sim's 28.11 m matches our path only if our path is started at tick 5 (t0 − 1.5 s; 28.138 m), and our arc over the first 0.5 s is 3.766 m. So the sim's 2 Hz window for this token appears to start 0.5 s later than its stated t0 − 2 s origin. The log is `2021.09.09.14.18.22_veh-48_00322_00895`, which is not one of the two logs with a missing frame in the selection-confirmation census (that census covers other tokens), so a missing frame is possible but not shown. The first sample timestamps on the sim side decide it.

`15715849` is not a second case: its 1.90 m is the 1.85 m move of the line to the connector start.

`MAP_FIELDS_A6` remains held for `090d6a17` only, because `stop_line_s` is a station measured from a path origin, and both sides must use the same origin.
