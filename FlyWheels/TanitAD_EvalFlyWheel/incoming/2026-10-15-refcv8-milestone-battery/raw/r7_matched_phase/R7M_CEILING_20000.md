# Does the speed-ceiling filter move the sharp-turn heading slope? (S2, CPU, banked dumps)

* generated 2026-10-18T06:08:45; B 2000; estimator: episode-cluster bootstrap, models FIXED (not another training run; not another inference run); tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* ON = the plan as banked in the panel (filter acting, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, filter OFF (`filter_off.traj`); REF = the banked refcv7-50,400 arm embedded in the panel (its filter was INERT: filter-on == filter-off in 4,754 / 4,754 windows).
* slopes are through-origin (turn-aligned heading error on turn-aligned GT heading); negative = turns less than GT; paired CIs; CIs answer 'another draw of EPISODES' only.

## R7-20k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.000 of 4754 windows, 0.0 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 365 / 365 | -0.0972 | -0.1055 | -0.1055 | -0.0083 [-0.0490, +0.0335] | -0.0083 [-0.0490, +0.0335] | +0.0000 [+0.0000, +0.0000] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1581 / 1581 | -0.0892 | -0.0997 | -0.0997 | -0.0106 [-0.0458, +0.0239] | -0.0106 [-0.0458, +0.0239] | +0.0000 [+0.0000, +0.0000] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 368 / 1098 | -0.1086 | -0.1057 | -0.1057 | +0.0029 [-0.0393, +0.0467] | +0.0029 [-0.0393, +0.0467] | +0.0000 [+0.0000, +0.0000] |

## EMA8-20k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.061 of 4754 windows, 0.09239130434782608 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 364 | -0.0968 | -0.1405 | -0.1370 | -0.0438 [-0.0750, -0.0180] SEP | -0.0402 [-0.0715, -0.0139] SEP | +0.0035 [-0.0004, +0.0082] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1577 / 1577 | -0.0883 | -0.1291 | -0.1267 | -0.0408 [-0.0682, -0.0172] SEP | -0.0384 [-0.0659, -0.0148] SEP | +0.0024 [-0.0015, +0.0066] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 367 / 1096 | -0.1084 | -0.1517 | -0.1491 | -0.0433 [-0.0726, -0.0182] SEP | -0.0408 [-0.0704, -0.0151] SEP | +0.0026 [-0.0007, +0.0065] |

## RAW8-20k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.093 of 4754 windows, 0.09239130434782608 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.1171 | -0.1110 | -0.0203 [-0.0653, +0.0173] | -0.0142 [-0.0592, +0.0245] | +0.0061 [-0.0008, +0.0148] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1569 / 1569 | -0.0872 | -0.1000 | -0.0949 | -0.0128 [-0.0512, +0.0210] | -0.0077 [-0.0459, +0.0261] | +0.0051 [-0.0009, +0.0125] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1093 | -0.1084 | -0.1260 | -0.1205 | -0.0176 [-0.0608, +0.0183] | -0.0121 [-0.0554, +0.0244] | +0.0055 [-0.0007, +0.0135] |

## Contrast X:Y = EMA8-20k:R7-20k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 364): X-Y ON -0.0356 [-0.0696, -0.0053] SEP; X-Y OFF -0.0320 [-0.0657, -0.0017] SEP; Y-REF ON -0.0082 [-0.0490, +0.0336]; Y-REF OFF -0.0082 [-0.0490, +0.0336]; X-REF ON -0.0438 [-0.0750, -0.0180] SEP; X-REF OFF -0.0402 [-0.0715, -0.0139] SEP
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1576): X-Y ON -0.0305 [-0.0596, -0.0041] SEP; X-Y OFF -0.0281 [-0.0583, -0.0020] SEP; Y-REF ON -0.0103 [-0.0453, +0.0242]; Y-REF OFF -0.0103 [-0.0453, +0.0242]; X-REF ON -0.0408 [-0.0682, -0.0172] SEP; X-REF OFF -0.0384 [-0.0659, -0.0148] SEP

## Contrast X:Y = RAW8-20k:R7-20k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 363): X-Y ON -0.0120 [-0.0528, +0.0226]; X-Y OFF -0.0060 [-0.0473, +0.0281]; Y-REF ON -0.0082 [-0.0490, +0.0336]; Y-REF OFF -0.0082 [-0.0490, +0.0336]; X-REF ON -0.0203 [-0.0653, +0.0173]; X-REF OFF -0.0142 [-0.0592, +0.0245]
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1568): X-Y ON -0.0027 [-0.0390, +0.0285]; X-Y OFF +0.0024 [-0.0339, +0.0336]; Y-REF ON -0.0101 [-0.0454, +0.0243]; Y-REF OFF -0.0101 [-0.0454, +0.0243]; X-REF ON -0.0128 [-0.0512, +0.0210]; X-REF OFF -0.0077 [-0.0459, +0.0261]
