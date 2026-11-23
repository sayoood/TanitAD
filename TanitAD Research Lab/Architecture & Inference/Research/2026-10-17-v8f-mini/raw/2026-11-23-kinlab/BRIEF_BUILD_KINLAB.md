# Brief: KINLAB phase 2, build `--dagger-label kin_v1` (the kinematic label of the ROLLHIST line). Zero training; the MM designs, you build and measure

You are a TanitAD build agent working for the Master Mind (MM). First read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (as in phase 1), then read your phase-1 deliverables:
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB.md`
* `F:/TanitAD-scratch/kinlab/HANDOFF_KINLAB.md`
* `F:/TanitAD-scratch/kinlab/KINLAB_CENSUS.md`

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands the code.
* No GPU.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* Never run anything on the LoRA pod. It is training until about 06:00Z. Read-only pulls of at most 1 GB are allowed.
* Work in a scratch export of **df250a00**: `F:/TanitAD-scratch/kinlab/tree`, made by `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive df250a00 stack taniteval | tar x -C ...`.
* The deliverable is the changed files, plus `CHANGED_FILES.txt` with their sha256 and the base blob of each. The MM lands them as one commit on `agent/v8f-rollhist-cdb-20261120`.

## Why (MEASURED, phase 1)
In an own-state sample, the inputs carry the own kinematics:
* `kin3` = (v, a, kappa), with `p10f` derived from it;
* `base` = `kdx_base(kin3)`, which is **O's prior path**.

The plan label `gt` follows the human path with the human's heading and curvature from NOW. `speed_v1` changes only the arc timing.

On RH bank s0, shard 0 (n = 372,791):
* yaw-rate gap at NOW: p50 1.84, p95 23.5 deg/s, against a human-to-human p95 of 1.78;
* acceleration gap: p50 1.55, p95 6.42 m/s², against a human-to-human p95 of 0.58.

So the label contradicts the prior path the model is given, and the own samples teach a large correction away from it. That is the interference reading left after ROLLADD's MECH2-NO. The kinematic label makes the label start where the inputs say the car is going, then merges it into the human's.

## The design (MM, binding; ask before deviating)
New value `kin_v1` of `--dagger-label`, next to `human` (default) and `speed_v1`.
* Both existing values stay **bit for bit** as at df250a00: gold tests against the df250a00 code on planted and real records.
* `kin_v1` is built per matcher-ok record. Off-route, end-edge and non-finite records are dropped exactly as today, and the eligible index is equal under all three values.

1. **Own start:** (v_o, a_o, kappa_o) = the record's `kin3` at NOW, the exact values the model reads (`p10f[-1, 0:3]` / `p10_t`). The pose is the origin and the heading is 0, in the sample frame (the logged frame at r', as re-anchored).
2. **Reference:** the record's `speed_v1` label, i.e. its per-step speeds u_l[k] and its point sequence, the human polyline at the speed_v1 arc. If `speed_v1` refuses the record, `kin_v1` refuses it too.
3. **Integration:** a unicycle at sub-step h = 0.05 s for 6.0 s (120 sub-steps). Output every 0.2 s (30 points) as `gt`; `s_h` is re-derived as `speed_v1` does.
   * Blend weight w(t) = min(1, t / T_B), with **T_B = 1.0 s**.
   * **Longitudinal:** a(t) = (1 − w)·a_o + w·a_ref(t), where a_ref is the speed_v1 per-step acceleration (u_l[k+1] − u_l[k]) / 0.2 for the step containing t. Clip |a| ≤ `A_LIM` (`v8f_recovery.py`, 3.96). v stays ≥ 0.
   * **Lateral:** kappa(t) = (1 − w)·kappa_o + w·kappa_pp(t).
     * kappa_pp is pure pursuit toward the reference polyline: the lookahead point is at arc s* + L_d along the reference from the closest point s*, with L_d = max(4.0 m, 1.0 s · v), and kappa_pp = 2·sin(alpha) / L_d.
     * Clip |kappa| ≤ `K_LIM` (0.198), and clip the friction term so that hypot(a, v²·kappa) ≤ `F_LIM` (6.80), scaling kappa down first.
4. **Merge:** once w = 1, |lateral error to the reference| < 0.10 m and |heading error| < 1.0°, the remaining points are the speed_v1 points from the corresponding reference arc on.
5. **Refusal:** today's fallback (the speed_v1 label, else the human label) is kept when:
   * any value is non-finite;
   * the label is not merged by t = 6.0 s;
   * the arc runs past the clip's logged path;
   * RV's feasibility checker (`violations`) flags the label.

   Count each reason separately.
6. **Identity rule:** if |kappa_o − kappa_ref0| ≤ 1e-6 and |a_o − a_ref0| ≤ 1e-6, the record's label is the speed_v1 label bit for bit, where kappa_ref0 / a_ref0 are the reference's first-step curvature and acceleration.
7. **Constants:** T_B, L_d's terms, the merge tolerances and h are module constants with a docstring that cites this brief. No CLI flag beyond the new value.

## Build
* Library: `stack/tanitad/data/v8f_recovery.py`, a new `kin_label_bank(...)` beside `label_bank`, reusing its matcher and checker.
* Trainer: the flag value in `stack/scripts/v8f_mini_train.py`, through FLAG_CHECKS / `built_levers()` exactly as `speed_v1` was wired at cf27dbb7.
* Bank: `attach_recovery`, `label_kind` returning "kin_v1".
* Counters: the trainer's `n_label_replaced` and a new `n_label_kin`, plus per-reason refusal counters in the selfcheck `dagger_probe`.
* Tests (new file `stack/tests/test_v8f_kinlab.py`, whole-file runs on the dev box):
  * gold: human and speed_v1 bit for bit vs df250a00;
  * the identity rule;
  * a planted record with own kappa > 0 must curve toward the own side, then merge;
  * a planted record with a_o > A_LIM must be refused for its reason;
  * the eligible index is equal under all three values;
  * the reachability chain flag → argparse → bank → swap → `gt` in the batch → the plan loss term. Assert a non-zero gradient difference between `kin_v1` and `speed_v1` on a planted batch, and an exact 0.0 for `speed_v1` against itself.
* Mutants (expect each RED): T_B = 0; w ignored; kappa_o ignored; a_o ignored; the merge never happens; the refusal falls back to human instead of speed_v1; the identity rule off; the friction clip off; the flag parsed but not passed to the bank.
* **Census** (`census_kinlab2.py`, CPU, dev box, shard 0 of RH bank s0, pulled read-only as in phase 1 and deleted afterwards):
  * share relabelled / identity / refused by reason, of ok records, per tick and pooled;
  * merge time p50 / p95;
  * max lateral deviation from the speed_v1 label p50 / p95 / p99;
  * |L_kin[30] − L_speed[30]| p50 / p95;
  * the same for the first 1 s.

  The MM reads this before freezing the card. ⚠️ Do not tune any constant on it. If the relabelled share is below 0.5, report it; do not change the design.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time.
* Gate each run with `wait_commit.py 5.5 60` (`C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`). Use 4.0 for a run under 1.5 GB of RAM if the gate stays shut, and say so.
* ⛔ Nothing on C:. TMP and TEMP go to `F:/TanitAD-scratch/kinlab/tmp`; pytest's `--basetemp` goes under it as well.
* F: is at about 8.4 GB free. Keep your peak under 1.5 GB and your folder under 600 MB after cleanup (the tree is ~450 MB). If F: drops below 6 GB, stop and hand back.
* Never kill a process you did not start, and never use a blanket `taskkill` or `pkill`.
* Never wait more than about 4 minutes in one call. If your context passes about 150k, write `HANDOFF_KINLAB2.md` and end.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* the changed files under `tree/`, listed in `CHANGED_FILES.txt` (path, sha256, base blob at df250a00);
* `FACTS_KINLAB_2.md` (where each piece is wired, with file:line);
* test logs, `mutants/MUTANTS_RESULT.json`;
* `census_kinlab2.py`, `KINLAB_CENSUS_2.json` and `.md`;
* `HANDOFF_KINLAB2.md`, `MANIFEST_SHA256.txt`, `DELETIONS.log`.

Hand back in at most 250 words, with a manifest, saying plainly what passed and what did not run.
