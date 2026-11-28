# Evidence, 2026-09-21 — rank distinctness and the distortion ablation

## 1 · The undistort ablation reproduces the OLD path EXACTLY

Checked against an **independently written** reimplementation of the pre-2026-09-21 formula
(`x = (u - cx)/fx * d`, then the extrinsic lift), NOT by re-running the model with a flag --
re-running a producer's own derivation measures determinism, not correctness.

```
undistort=False  vs the independent rewrite : max |delta| 0.000e+00   <- bit-identical
undistort=True   vs the same reference      : max |delta| 1.486e-01
shapes (1, 7680, 6)
ABLATION_REPRODUCES_OLD_BEHAVIOUR
```

And the inversion itself is verified against an **analytic target**: ideal -> distorted (forward
Caltech) -> our fixed-point inverse -> ideal, worst error **1.581e-13** over 15 field points.
HFOV moves **62.85 deg -> 70.92 deg** at the last patch centre.

## 2 · The ranks are genuinely distinct, and the control proves it

⛔ Before the fix (rank 1 built from the rank-0 rollout dir, `--rank` being only a LABEL):
target bank **0 of 1,091** keys differed in trajectory, goal or image; scorer bank **0 of 2,332**
trajectories and **0 of 2,332** PDM targets differed. Files had EXACTLY the same byte count
(6,302,277) with DIFFERENT md5s, so neither a size nor a hash check could see it.

After rebuilding rank 1 from `C:/dzo/m-nr-r1`:

| bank | measure | result |
|---|---|---|
| targets | trajectories differing | **1,091 / 1,091** |
| targets | goals differing | 1,064 / 1,091 (median separation **3.57 m**) |
| targets | images differing | **0 / 1,091** (correct -- same frames, different route) |
| targets | true duplicates (same traj AND goal) | **0** |
| scorer | route-DEPENDENT candidates identical | **0 / 1,791** |
| scorer | route-INDEPENDENT (`stopped`) identical | **199 / 199** |
| scorer | PDM targets differing | 2,183 / 2,189 |
| scorer | teacher-off-road frames | rank0 **22**, rank1 **132** |

⭐ **`stopped` is the positive control and it is the strongest line in the table.** A zero-velocity
plan cannot depend on the route, so it MUST be identical -- and it is, on all 199. The asymmetry
(everything route-dependent moved, the route-independent thing did not) is a far stronger claim
than "most rows differ". If `stopped` ever differs across ranks, something upstream is wrong.

⚠️ `over-curb` coincides on 5 of 199: it aims at the NEAREST curb, which can be the same from both
routes. Listed as MAY_COINCIDE rather than silently tolerated.

## 3 · Scorer coverage: the number did not change, its MEANING did

Before: "19.4 %" = the same 212 frames counted twice under two labels.
After:  **19.4 % = 424 genuinely distinct covered frames** over 2,182 genuinely distinct tuples.
A figure that survives its own correction unchanged is exactly the kind that escapes review.

Instrument: `refe/diag_rank_distinctness.py --self-test --scorer <r0.jsonl> <r1.jsonl>`
Verdict: `RANKS_ARE_DISTINCT` + `SELF_TEST_OK`.
