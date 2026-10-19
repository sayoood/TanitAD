# Does the speed-ceiling filter move the sharp-turn heading slope? (S2, CPU, banked dumps)

* generated 2026-10-18T03:49:33; B 2000; estimator: episode-cluster bootstrap, models FIXED (not another training run; not another inference run); tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* ON = the plan as banked in the panel (filter acting, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, filter OFF (`filter_off.traj`); REF = the banked refcv7-50,400 arm embedded in the panel (its filter was INERT: filter-on == filter-off in 4,754 / 4,754 windows).
* slopes are through-origin (turn-aligned heading error on turn-aligned GT heading); negative = turns less than GT; paired CIs; CIs answer 'another draw of EPISODES' only.

## EMA8-5k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.067 of 4754 windows, 0.10869565217391304 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 364 | -0.0968 | -0.1316 | -0.1235 | -0.0348 [-0.0584, -0.0162] SEP | -0.0268 [-0.0510, -0.0061] SEP | +0.0080 [+0.0034, +0.0136] SEP |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1579 / 1579 | -0.0887 | -0.1204 | -0.1147 | -0.0317 [-0.0518, -0.0151] SEP | -0.0260 [-0.0462, -0.0083] SEP | +0.0057 [+0.0004, +0.0112] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 367 / 1096 | -0.1084 | -0.1438 | -0.1369 | -0.0354 [-0.0568, -0.0175] SEP | -0.0285 [-0.0508, -0.0086] SEP | +0.0069 [+0.0029, +0.0117] SEP |

## EMA8-10.5k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.061 of 4754 windows, 0.08423913043478261 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 364 | -0.0968 | -0.1570 | -0.1518 | -0.0602 [-0.0941, -0.0322] SEP | -0.0550 [-0.0889, -0.0272] SEP | +0.0052 [+0.0018, +0.0095] SEP |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1577 / 1577 | -0.0883 | -0.1472 | -0.1421 | -0.0589 [-0.0880, -0.0346] SEP | -0.0538 [-0.0820, -0.0302] SEP | +0.0051 [+0.0012, +0.0098] SEP |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 367 / 1096 | -0.1084 | -0.1674 | -0.1631 | -0.0590 [-0.0919, -0.0310] SEP | -0.0547 [-0.0869, -0.0269] SEP | +0.0042 [+0.0013, +0.0081] SEP |

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

## RAW8-5k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.081 of 4754 windows, 0.07065217391304347 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 361 / 361 | -0.0953 | -0.2323 | -0.2310 | -0.1370 [-0.1914, -0.0887] SEP | -0.1357 [-0.1907, -0.0883] SEP | +0.0013 [-0.0028, +0.0052] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1571 / 1571 | -0.0870 | -0.2110 | -0.2099 | -0.1240 [-0.1701, -0.0814] SEP | -0.1230 [-0.1700, -0.0799] SEP | +0.0010 [-0.0029, +0.0046] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 364 / 1088 | -0.1071 | -0.2485 | -0.2475 | -0.1414 [-0.1957, -0.0933] SEP | -0.1404 [-0.1938, -0.0926] SEP | +0.0011 [-0.0031, +0.0049] |

## RAW8-10.5k

* control s0: dump `os` vs `nav_true.traj[:, :4]` max abs 0.0e+00; panel `os` vs it 0.0e+00 (both must be 0); filter changes the selected anchor on 0.085 of 4754 windows, 0.13858695652173914 of the 368 sharp windows

### sharp (>= 0.15) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.1380 | -0.1240 | -0.0412 [-0.1109, +0.0145] | -0.0272 [-0.0802, +0.0234] | +0.0140 [-0.0015, +0.0402] |

### curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 1572 / 1572 | -0.0882 | -0.1247 | -0.1141 | -0.0365 [-0.0951, +0.0118] | -0.0258 [-0.0710, +0.0185] | +0.0107 [-0.0024, +0.0311] |

### sharp (>= 0.15) | segments 2..K

| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |
|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1092 | -0.1083 | -0.1454 | -0.1332 | -0.0371 [-0.1044, +0.0165] | -0.0248 [-0.0742, +0.0228] | +0.0122 [-0.0032, +0.0379] |

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
