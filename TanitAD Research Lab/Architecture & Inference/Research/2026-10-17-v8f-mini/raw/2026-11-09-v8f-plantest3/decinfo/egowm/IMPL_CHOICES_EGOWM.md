# IMPL_CHOICES_EGOWM (card: PREREG_V8F_EGOWM.md, frozen sha256 ddd6fc9b...f446e7)

Honesty note: the design choices below were fixed before the code was written; the file:line trace in section 2 was filled in after the patch (`egowm.diff`, sha256 in `egowm.diff.sha256`). Lines refer to the patched tree `F:/TanitAD-scratch/egowm/tree` (rebuild: `git apply egowm.diff` on commit 1faa1a06).

## 1. Choices
* **Flag:** `--wm-speed {off,kin,now}`, default `off`. Config field `V8FMiniConfig.wm_speed`. Off builds the control (control 1: 579 of 579 arrays bit-equal to the baseline tree).
* **Where the speed enters:** an added action-embedding channel per level, `spd[l]` = bias-free `Linear(1, E_l)` (E = 32, 32, 32), zero-initialised, input `speed / WM_SPEED_SCALE` (10 m/s). Level 1: added to `act_embed(a)`. Levels 2 and 3: added to the macro-action embedding that the level's predictor reads. The macro encoders pool the speed-free level-1 embeddings, so each level has its own speed parameters (control 2 reads each one).
* **Built on a forked RNG**, so the global stream and every other parameter equal the control's (control 1b: all tensors bit-equal at init, all 25 loss terms bit-equal with zero weights).
* **Speed normalisation:** v / 10 m/s. Step time of one level-1 step: 0.2 s (`WM_STEP_S`, the cache grid). Integration floor: applied at every level-1 step (closed form `c - min(0, running min c)`, checked against a loop).
* **Measured speed at NOW on the `--wm-zcache` path:** the cached path has no frames, but `item()` already reads the clip kinematics through `_kin_small` (`v8f_wm_windows.py`). `_kin_small` now also keeps `kin3[:, 0]` (v0, the measured speed from the native pose track in yaw mode, the ego state S, T, O read). `item()` returns `v` [F] for every window frame (a window holding a NaN speed, i.e. clip row 0, is refused, never zero-filled). The labelled probe windows get the same `v` through `V8FMiniWindows(with_v=True)`.
* **What NOW means (decision, flagged):** NOW is the **anchor of the free run** (level-1 state i of the k-step loss, frame i x cum_stride of the level), because the planner starts every rollout at its NOW. Teacher-forced one-step terms read the measured speed of the step input frame (a rollout of length 1 from its own anchor). The k-step terms (`--wm-rollout-k 10 10 6`) read `anchored_speeds`: E-NOW = measured speed at the anchor held over the k steps; E-KIN = that speed integrated over the level-1 accelerations (`act[..., 0]`, m/s^2) of the frames the run covers, floored at 0. A single per-window NOW (first or last frame of 91 frames) was rejected: it would hold an 18 s old speed for late anchors in E-NOW, or leak a later speed into earlier steps.
* **Goal and target latents stay vision-only:** `encode` adds speed to the action embeddings only; `enc.z` is untouched (control 4).
* **Planner harness at inference (not part of this patch; for the Master Mind):** the harness must call the speed stack with a speed per step. `EgoHJEPAStack.rollout(level, z0, actions, context=, speed=[..., H])` takes it; E-NOW: `speed = v_now` repeated; E-KIN: `reflect_integrate(v_now, candidate_acc)` of the CANDIDATE actions (differentiable, so the planner gradient sees it). `ladder_plantest/lp_lib.py::rollout_h` calls `st.act_embed` directly and must add `st.speed_embed(1, speed)` to the embedding; a speed arm raises if no speed is given. Context-3 history rows use the measured speed of their frames. This harness change is NOT written.
* **Fourth file:** the card names three files; the zcache dataset lives in `tanitad/data/v8f_wm_windows.py`, so the patch touches that fourth file too. `hjepa.py`, `models/__init__.py`, `launch_gate.py` are not edited (the speed stack is a subclass in `v8f_mini.py`).

## 2. Trace (argparse -> namespace -> model attribute -> forward -> loss term)
| hop | where |
|---|---|
| argparse `--wm-speed` | `v8f_mini_train.py:318` |
| gate entry (every flag needs one) | `v8f_mini_train.py:123` |
| namespace -> config (the one line, mutated in control 5) | `v8f_mini_train.py:418` `wm_speed=a.wm_speed` |
| config field and validation | `v8f_mini.py:346`, `431-434` |
| model attribute: stack class chosen | `v8f_mini.py:844-845` (`self.speed`, `ego_stack_cls()`); spd modules built in `EgoHJEPAStack.__init__` (class at `v8f_mini.py:207`) |
| data: windows return `v` | `v8f_wm_windows.py:50` (ctor `ego`), `120` (item), collate `v`; trainer passes `ego=` at `v8f_mini_train.py:3073`; record at `3086` |
| forward: batch `v` to loss | `v8f_mini.py` `wm_only_loss` passes `batch.get("v")` to `forward_mean` (927), then `ego_of` (909), then `stack.loss(..., ego=)` |
| teacher-forced term | `EgoHJEPAStack.encode` (242): `enc.act[i] + speed_embed(i+1, vm[:, 0:(n-1)*cum:cum])`, then `_pred_loss` |
| k-step term | `EgoHJEPAStack.rollout_loss` (256): `anchored_speeds` (180), then `rollout` (268): `e + speed_embed(level, speed)`, giving `rollout/L{l}` in the level loss |
| read-back (control 5) | `WMAux.built_speed` (930) classifies the BUILT rule on a planted ramp; `bad_wm_speed` (`v8f_mini_train.py:1018`) wired at `1463`; selfcheck record at `3239` |

## 3. Known limits (UNVERIFIED)
* `--selfcheck-only` and the full G-DVB path on real data were not run (the cache lives on box 2); `bad_wm_speed` was run on built models with a stand-in data record.
* How well integrating `act[..., 0]` x 0.2 s reproduces the measured speed over a rollout was not measured (the available val file carries no speed). E-KIN uses the actions as the card says.
* The repository test suite (`stack/tests`) was not exported or run.
* The seed-2 replicate keeps the readers and zcache of M2b_s0 (no M2b_s2 is known to exist).
* The only supported path is `--wm-speed` with `--wm-only --wm-windows wmonly`; the full-model loss path raises if a speed arm gets no `v`.
* The box-2 chain (deliverable 5) was not written.
