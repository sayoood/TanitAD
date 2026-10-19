# Does the speed-ceiling filter move the sharp-turn heading slope? (S2, CPU, banked dumps)

* generated 2026-10-19T18:27:51; B 2000; estimator: episode-cluster bootstrap, models FIXED (not another training run; not another inference run); tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* ON = the plan as banked in the panel (filter acting, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, filter OFF (`filter_off.traj`); REF = the banked refcv7-50,400 arm embedded in the panel (its filter was INERT: filter-on == filter-off in 4,754 / 4,754 windows).
* slopes are through-origin (turn-aligned heading error on turn-aligned GT heading); negative = turns less than GT; paired CIs; CIs answer 'another draw of EPISODES' only.

## R7-15k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.000 of 4754 windows, 0.0 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 364 | -0.0968 | -0.2344 | -0.2344 | -0.1376 [-0.1891, -0.0908] SEP | -0.1376 [-0.1891, -0.0908] SEP | +0.0000 [+0.0000, +0.0000] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1580 / 1580 | -0.0888 | -0.2176 | -0.2176 | -0.1288 [-0.1725, -0.0869] SEP | -0.1288 [-0.1725, -0.0869] SEP | +0.0000 [+0.0000, +0.0000] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 367 / 1097 | -0.1084 | -0.2288 | -0.2288 | -0.1205 [-0.1692, -0.0745] SEP | -0.1205 [-0.1692, -0.0745] SEP | +0.0000 [+0.0000, +0.0000] |

## EMA8-15k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.059 of 4754 windows, 0.08423913043478261 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.1397 | -0.1370 | -0.0429 [-0.0791, -0.0130] SEP | -0.0402 [-0.0770, -0.0092] SEP | +0.0027 [-0.0015, +0.0080] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1576 / 1576 | -0.0883 | -0.1323 | -0.1289 | -0.0440 [-0.0749, -0.0190] SEP | -0.0406 [-0.0714, -0.0142] SEP | +0.0034 [+0.0000, +0.0078] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1094 | -0.1084 | -0.1512 | -0.1497 | -0.0428 [-0.0771, -0.0141] SEP | -0.0414 [-0.0756, -0.0115] SEP | +0.0015 [-0.0025, +0.0061] |

## RAW8-15k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.055 of 4754 windows, 0.059782608695652176 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 361 / 361 | -0.0954 | -0.2187 | -0.2146 | -0.1233 [-0.1752, -0.0861] SEP | -0.1192 [-0.1713, -0.0820] SEP | +0.0040 [+0.0010, +0.0079] SEP |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1564 / 1564 | -0.0861 | -0.1984 | -0.1945 | -0.1123 [-0.1563, -0.0782] SEP | -0.1084 [-0.1524, -0.0743] SEP | +0.0039 [+0.0009, +0.0074] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 1086 | -0.1068 | -0.2235 | -0.2199 | -0.1168 [-0.1661, -0.0812] SEP | -0.1131 [-0.1628, -0.0773] SEP | +0.0037 [+0.0010, +0.0072] SEP |

## Contrast X:Y = EMA8-15k:R7-15k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 363): X-Y ON +0.0939 [+0.0520, +0.1386] SEP; X-Y OFF +0.0966 [+0.0551, +0.1409] SEP; Y-REF ON -0.1368 [-0.1870, -0.0905] SEP; Y-REF OFF -0.1368 [-0.1870, -0.0905] SEP; X-REF ON -0.0429 [-0.0791, -0.0130] SEP; X-REF OFF -0.0402 [-0.0770, -0.0092] SEP
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1575): X-Y ON +0.0840 [+0.0483, +0.1225] SEP; X-Y OFF +0.0875 [+0.0507, +0.1254] SEP; Y-REF ON -0.1281 [-0.1709, -0.0868] SEP; Y-REF OFF -0.1281 [-0.1709, -0.0868] SEP; X-REF ON -0.0440 [-0.0750, -0.0190] SEP; X-REF OFF -0.0406 [-0.0715, -0.0142] SEP

## Contrast X:Y = RAW8-15k:R7-15k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 361): X-Y ON +0.0131 [-0.0471, +0.0735]; X-Y OFF +0.0171 [-0.0433, +0.0774]; Y-REF ON -0.1364 [-0.1872, -0.0904] SEP; Y-REF OFF -0.1364 [-0.1872, -0.0904] SEP; X-REF ON -0.1233 [-0.1752, -0.0861] SEP; X-REF OFF -0.1192 [-0.1713, -0.0820] SEP
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1563): X-Y ON +0.0156 [-0.0357, +0.0693]; X-Y OFF +0.0195 [-0.0319, +0.0728]; Y-REF ON -0.1279 [-0.1702, -0.0870] SEP; Y-REF OFF -0.1279 [-0.1702, -0.0870] SEP; X-REF ON -0.1123 [-0.1563, -0.0782] SEP; X-REF OFF -0.1084 [-0.1524, -0.0742] SEP
