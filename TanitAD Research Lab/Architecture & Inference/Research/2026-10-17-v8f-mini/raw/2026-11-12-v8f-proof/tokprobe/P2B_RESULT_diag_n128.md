# P2B_RESULT: control p2b of card ADD27 (note N1 section 2)

Tier T0 (a diagnostic on logged frames, never driving performance). Evidence class: MEASURED (synthetic). STRATEGIC: PAUSED (PI item 25).

World: tokens = noise + 6*(gap-30)/10 * U[k]; no closing-speed direction; gap = g0 + c*0.2*frame (the gap moves with the closing speed); 128 train / 8 held-out clips of 24 frames.

| arm | steps | held-out R2 gap | held-out R2 closing | both above 0.9 |
|---|---|---|---|---|
| TK16_s0 | 500 | +0.915 | -0.006 | no |
| TK16_s0 | 2000 | +0.767 | +0.005 | no |
| TK16_s0 | 5000 | +0.489 | -0.563 | no |

p2b passes for an arm if both R2 are above 0.9 at one of the step counts. Per arm: {"TK16_s0": false}. p2b overall: FAIL.

Oracle readability of this plant (not the probe): R2 gap +0.999, R2 closing +0.936 on n=60 held-out windows.

