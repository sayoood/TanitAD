# The 3-D agent join reaches the box head — the last hop, closed

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: dev box, CPU**
**Closes:** the one item `24065b6` shipped open.

## What was open

The 3-D agent join landed in `24065b6`. Its author flagged one hop they could
not check, and I repeated the flag in the commit message:

> `box3d_head.py` does not exist on this branch. The last hop
> (`3-D join → zh_targets → n_z > 0`) is therefore **UNVERIFIED** here.

Their worktree predated `box3d_head.py`. The head is now landed, so the check
runs — and it is run here, against the artifact itself, not a fixture.

⚠️ **My first probe failed on itself, not on the data.** It asserted
`"center_z" in ag and "size_z" in ag`. Those are the **parquet's** names. The
join's own names are **`cz`** and **`h`**. The join was correct throughout; the
probe was wrong. The corrected docstring in `box3d_head.py` now says so
explicitly, because the next reader will reach for the parquet's names too.

## What was measured

Artifact: `b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz`
(md5 `ff0e68fd41e86f6a30180ff6173b0685`), 139 clips, 26,394 index entries,
905,512 agents. Fields per agent:
`cls, cx, cy, cz, h, l, occ, track_id, w, yaw`.

| hop | call | result |
|---|---|---|
| open | `open_join3d` | 139 clips / 26,394 lines / 905,512 agents |
| read | `zh_for_frame` | 22 / 22 tracks on the probe frame |
| target | `zh_targets` | **n_z = 22**, cz −0.015 … +2.117 m, h 1.433 … 3.561 m |
| geometry | `base_from_centre` | bottom faces a median **−0.103 m** off the ego ground plane |
| loss | `box3d_set_loss` | **n["z"] = n["h"] = 22 > 0**, `loss_z` 1.5268, `loss_h` 0.8218 |
| control | same prediction, labels withheld | `n["z"] = 0`, `loss_z` **exactly 0.0**, total 89.5158 |
| separation | with − without | **+2.3486** |
| coverage | every line in the file | **905,512 / 905,512 agent-frames (100.00 %)** |

The **−0.103 m** median matters on its own: it is what shows `cz`/`h` are a
centre-and-extent pair, not a bottom-and-top pair. A bottom/top reading would
put the median near `−h/2 ≈ −0.9 m`.

591 of the 26,394 lines carry **no agents at all**. That is LABELLED CLEAR, a
state of its own, and it is excluded from the coverage denominator rather than
counted as a miss — the join doc's §4 rule.

## Why this is a guard and not an inspection

Both tests were run three ways:

| run | outcome |
|---|---|
| on the artifact | **2 passed** |
| `$TANITAD_AGENT_JOIN3D` unset | **2 skipped**, with the reason printed |
| MUTATION: `AgentJoin3D.zh` returns an all-False mask | **2 failed** |

The mutation is the one that earns the landing. Under it the coverage test
reads `905512 of 905512 agent-frames over 25803 lines carry no z/h` — so the
assertion is reachable, and a join that silently stops answering cannot pass.

## The record defects this turned up

1. `box3d_head.py` cited **`tests/test_refcv6_box3d.py`** three times. **That
   file does not exist.** Both cited tests are real
   (`test_no_zh_labels_is_the_2d_loss_exactly`,
   `test_AP_of_perfect_predictions_is_exactly_one`) — they live in
   `tests/test_refcv6_perception.py`. A citation that resolves to nothing turns
   "asserted by a test" into an unverifiable claim. All three corrected.
2. The same docstring stated the eval join carries **"no z, no h"**. That was
   MEASURED and true on 2026-09-16 and became wrong on 2026-09-17. Corrected
   with the supersession stamped, the old fact kept, and the `cz`/`h` versus
   `center_z`/`size_z` trap named.

## Status

⛔ Nothing here is a capability claim. It establishes that the **label path is
wired and carries real metres** — not that the head predicts height well. That
is a training result and has not been run.

`SPEC_REFCV6_V2.md` §6 said of the box head: *"Today's seam emits (cx, cy, l, w,
yaw): **add z and h**."* That is now done and measured end to end.

## Files

| path | what |
|---|---|
| `code/lasthop_zh_evidence.py` | the end-to-end probe, banked |
| `raw/lasthop_zh.json` | its measured output |
| `stack/tests/test_refcv6_perception_realdata.py` | the two permanent tests |
