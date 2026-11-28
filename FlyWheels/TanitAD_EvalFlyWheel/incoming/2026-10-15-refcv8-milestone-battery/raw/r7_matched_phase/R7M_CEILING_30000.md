# Does the speed-ceiling filter move the sharp-turn heading slope? (S2, CPU, banked dumps)

* generated 2026-10-18T23:14:23; B 2000; estimator: episode-cluster bootstrap, models FIXED (not another training run; not another inference run); tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* ON = the plan as banked in the panel (filter acting, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, filter OFF (`filter_off.traj`); REF = the banked refcv7-50,400 arm embedded in the panel (its filter was INERT: filter-on == filter-off in 4,754 / 4,754 windows).
* slopes are through-origin (turn-aligned heading error on turn-aligned GT heading); negative = turns less than GT; paired CIs; CIs answer 'another draw of EPISODES' only.

## R7-30k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.000 of 4754 windows, 0.0 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 365 / 365 | -0.0972 | -0.1389 | -0.1389 | -0.0417 [-0.0660, -0.0187] SEP | -0.0417 [-0.0660, -0.0187] SEP | +0.0000 [+0.0000, +0.0000] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1582 / 1582 | -0.0891 | -0.1259 | -0.1259 | -0.0367 [-0.0569, -0.0164] SEP | -0.0367 [-0.0569, -0.0164] SEP | +0.0000 [+0.0000, +0.0000] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 368 / 1098 | -0.1086 | -0.1409 | -0.1409 | -0.0322 [-0.0554, -0.0097] SEP | -0.0322 [-0.0554, -0.0097] SEP | +0.0000 [+0.0000, +0.0000] |

## EMA8-30k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.077 of 4754 windows, 0.10054347826086957 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.1301 | -0.1229 | -0.0333 [-0.0622, -0.0066] SEP | -0.0261 [-0.0552, +0.0014] | +0.0072 [+0.0003, +0.0161] SEP |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1577 / 1577 | -0.0887 | -0.1167 | -0.1105 | -0.0280 [-0.0542, -0.0040] SEP | -0.0218 [-0.0479, +0.0026] | +0.0062 [+0.0003, +0.0134] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1094 | -0.1084 | -0.1415 | -0.1351 | -0.0331 [-0.0594, -0.0075] SEP | -0.0267 [-0.0541, -0.0005] SEP | +0.0064 [+0.0002, +0.0145] SEP |

## RAW8-30k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.076 of 4754 windows, 0.11141304347826086 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 365 / 365 | -0.0972 | -0.1534 | -0.1438 | -0.0561 [-0.0971, -0.0171] SEP | -0.0466 [-0.0892, -0.0046] SEP | +0.0095 [+0.0020, +0.0193] SEP |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1574 / 1574 | -0.0880 | -0.1388 | -0.1288 | -0.0507 [-0.0890, -0.0130] SEP | -0.0408 [-0.0784, -0.0021] SEP | +0.0100 [+0.0033, +0.0180] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 368 / 1097 | -0.1086 | -0.1597 | -0.1507 | -0.0511 [-0.0891, -0.0142] SEP | -0.0421 [-0.0814, -0.0029] SEP | +0.0090 [+0.0022, +0.0181] SEP |

## Contrast X:Y = EMA8-30k:R7-30k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 363): X-Y ON +0.0084 [-0.0189, +0.0328]; X-Y OFF +0.0156 [-0.0116, +0.0415]; Y-REF ON -0.0417 [-0.0661, -0.0187] SEP; Y-REF OFF -0.0417 [-0.0661, -0.0187] SEP; X-REF ON -0.0333 [-0.0622, -0.0066] SEP; X-REF OFF -0.0261 [-0.0552, +0.0014]
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1577): X-Y ON +0.0087 [-0.0158, +0.0315]; X-Y OFF +0.0149 [-0.0100, +0.0390]; Y-REF ON -0.0366 [-0.0567, -0.0163] SEP; Y-REF OFF -0.0366 [-0.0567, -0.0163] SEP; X-REF ON -0.0280 [-0.0542, -0.0040] SEP; X-REF OFF -0.0218 [-0.0479, +0.0026]

## Contrast X:Y = RAW8-30k:R7-30k (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)

* seed 0 -- sharp (>= 0.15) | last segment (n 365): X-Y ON -0.0144 [-0.0510, +0.0235]; X-Y OFF -0.0049 [-0.0428, +0.0338]; Y-REF ON -0.0417 [-0.0660, -0.0187] SEP; Y-REF OFF -0.0417 [-0.0660, -0.0187] SEP; X-REF ON -0.0561 [-0.0971, -0.0171] SEP; X-REF OFF -0.0466 [-0.0892, -0.0046] SEP
* seed 0 -- curved (GT |yaw| >= 0.02) | last segment (n 1574): X-Y ON -0.0141 [-0.0482, +0.0234]; X-Y OFF -0.0041 [-0.0383, +0.0323]; Y-REF ON -0.0367 [-0.0567, -0.0163] SEP; Y-REF OFF -0.0367 [-0.0567, -0.0163] SEP; X-REF ON -0.0507 [-0.0890, -0.0130] SEP; X-REF OFF -0.0408 [-0.0784, -0.0021] SEP
