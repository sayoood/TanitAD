# HANDOFF: DISTRULE tool (card PREREG_V8F_PROOF_ADD26). Builder: teammate, dev-box CPU only. Nothing staged, committed, pushed; no pod, box 2 or GPU touched.

Tier T1-kin (kinematic rollout on the model's own actions, logged frames); diagnostic, never driving performance. No number of the card was computed on a real arm. Strategic family: PAUSED (PI item 25).

## State
All deliverables exist and the tests pass (22 passed, 59 s, CPU: `python -m pytest test_dist_rule.py -q --basetemp=<dir with existing parent>`). Everything is in `F:/TanitAD-scratch/v8proof/distrule/`.

## Two findings the lead must read first (details: DISTRULE_DEFS E3b, E5)
1. **The floor lifts plans.** Card form a' = max(a_f, min(a_w, a_c)): wherever a_f > a_w (plan slows faster than 6 m/s2, or its first step is >1.2 m/s below measured v0) the new plan is AHEAD of what the model issued, with or without any lead cap. On a 16-window CPU smoke of P_s0 (no card number): label rule changed 10 of 14 active ticks (55 lift steps), `never` (r2 constants) changed 8 ticks / 5 of 16 windows by lift alone. So **r2 as written cannot hold on real plans**. Builder option `--floor cap-only` (a' = min(a_w, max(a_f, a_c)); default `card` = as written): on the same smoke `never` + cap-only equals `off` bit for bit. `cap-only` is not the card; using it for arms needs your amendment. Every rule log has `lift` per tick.
2. **The r2 constants are not never-binding.** With TAU = 0, max(D, TAU*v') = 0, so the cap is the lead position (zero margin): it binds on contact windows. Mode `plumb` (cap and floor absent, whole wrapper path active) is a guaranteed identity probe, tested: active ticks > 0 and dump equal to `off`. Pick which one is read as r2.

## What was verified (file: test_dist_rule.py, all on synthetic worlds from `make_synthetic_world` plus a real-input smoke)
- r5: hand-built cases (stopped lead 10 m at 10 m/s: braking-limited profile, stop at 7.36 m, 30 saturated / 10 executed; faster lead: plan unchanged bit for bit; curved plan keeps its path); 300 random plans: progress never decreases, braking floor holds, lag form equals a card-literal scalar reference (1e-8).
- Inactive windows bit for bit; all-inactive returns the same array object. Progress geometry copy equals `headway_read` (`logged_arc`, `progress`, constants; subprocess because of the taniteval stack guard).
- Label source equals `lead_sup_label_arrays` row for row; r4 label: rows after the tick set to NaN change nothing (all ticks of all clips), with a firing control.
- r6 on the tiny-checkpoint head source: equals `lead_dump.head_outputs` within 1e-4 on >= 64 ticks (p, gap, closing); r4 head: noise in all tokens after the last tick changes no output, noise inside the frames does.
- End to end through the pinned `rollout` with a stub planner: no lead -> every mode equals the unwrapped rollout exactly; stopped lead -> label reduces progress, push increases it, plumb equals off while active, `never` changes at least one tick; windows without a lead frame untouched bit for bit; logged a_m equals `progress` of the rolled path in the dump.
- Mutations (rule computed but not applied; rule never active; wrapper planner returning the base plan) turn the reduction checks and the CLI reach test red.
- CLI: `--dist-rule off` equals the pinned tool called directly (dump bytes, tick file, file list); label and head reachable from argparse to the dump with a rule log; `--rule-log` with `off` is refused.
- Real-input CPU smoke (deliverable 5), checkpoint `C:/Users/Admin/v8proof_c/ckpt/P_s0/ckpt_last.pt` (no lead head, so label only), eval cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache`, tracks `.../sfresh_inputs/inputs/native_tracks.pt`, grid and `join_eval141.npz` from `E:/.../dagt1_tree_b43c7a9a/.../2026-10-17-v8f-mini/raw/`, `--max-windows 16`: control r1 holds (rule off through the wrapper equals the pinned tool: 16 dump files, tick file byte-identical; the json differs only in `ticks_file`); label, never, plumb, push, and the cap-only forms ran. Script `smoke_cpu.sh`, outputs `tmp/smoke/`. Not scored (t1_eval is not in this tree; `--no-analyze`).
- Pod script: refusals (bad mode, missing ckpt, <20 GB free via fake cgroup files, existing output), layout, one process, no overwrite, tested with a fake python (`bash -n` ok).

## Not verified
- r1 against the BANKED pod dump (needs the pod). r2, r3 by the reader, r7 (headway_read h1-h6, q3_read controls on the new dumps): not run. r6 on a real LS/LSH/LSD checkpoint and r4 on real arms: only the tiny-checkpoint forms ran (the tests also run on the pod CPU: `DISTRULE_TREE=<tree> python -m pytest test_dist_rule.py -k "r4 or r6"`; that adds CPU load, your call).
- The head source on a real checkpoint (none on the dev box). The pod join `join_eval141.npz` sha256 vs the dev-box one: the rule log stores its sha256; compare.
- Memory/time on the pod; the wrapper adds one numpy projection per tick (30 steps, 32 windows), negligible.

## Commands for the lead (pod)
Place `dist_rule.py`, `v8f_t1_eval_rule.py`, `pod_distrule_run.sh` in one folder (for example `/workspace/v8proof/distrule/`); tree `/workspace/v8proof/tree_leadsup` (pinned tool sha256 61e140ed..., the script refuses otherwise). `Q` = the queue folder of the base arm (`/workspace/v8proof/runs/Q/<BASE>/ckpt_last.pt`). Names: RL label, RH head, RN never, RO off, RP push, RM plumb. Optional `DR_FLOOR=cap-only`. One rollout per call, about 8 GPU-min.
```
S=/workspace/v8proof/distrule/pod_distrule_run.sh
# r1: off through the wrapper vs the banked dump of the same checkpoint (one arm), bit for bit
bash $S HIERRO_s0 Q HIER_s0 off ; for f in /workspace/v8proof/runs/distrule/t1/HIERRO_s0/dump_HIERRO_s0_m10_causal_heads/ep*.npz ; do b=$(basename $f); cmp $f /workspace/v8proof/runs/Q/t1/HIER_s0/dump_HIER_s0_m10_causal_heads/$b || echo DIFF $b; done
# (the npz files embed no name, but compare arrays with numpy if cmp differs on zip metadata)
# r2 (card constants) and r2' (plumb), r3
bash $S HIERRN_s0 Q HIER_s0 never ; bash $S HIERRM_s0 Q HIER_s0 plumb ; bash $S HIERRP_s0 Q HIER_s0 push
# arm A: label on HIER_s0..s5 and G_s0..s5 (12 rollouts; at most three at a time)
for A in HIER G ; do for s in 0 1 2 3 4 5 ; do bash $S ${A}RL_s$s Q ${A}_s$s label ; done ; done
# arm B: head and label on LS, LSH, LSD seeds 0 and 1 (12 rollouts), after ADD24
for A in LS LSH LSD ; do for s in 0 1 ; do bash $S ${A}RH_s$s Q ${A}_s$s head ; bash $S ${A}RL_s$s Q ${A}_s$s label ; done ; done
```
Pull on the dev box: `Q=distrule bash F:/TanitAD-scratch/v8proof/spatial/q3/pull_spatial_t1.sh HIERRL_s0 ...` (the rule log, run record `rule_run_<NEW>.txt` and ticks are inside `ticks/` and travel in the same tar). Note the pull script derives the dump folder name `<ARM>_m10_causal_heads` from the arm name, which the pod script uses. The comparison no-rule dump of a base arm is the banked one (`hiergap/pull/<BASE>`); `headway_read.py` reads the new arms as arms named `HIERRL` seeds 0..5 etc.

## Files that go to the pod (sha256)
- dist_rule.py 40b5dd9173aeaf8c7d7a7d4bfbb52484a8f30b9a8480d3356f8fa0aee65746dc
- v8f_t1_eval_rule.py 51e196758cd878fca261312ca09f53c1a57c0131b6b3c50bc56d5bd98c553a03
- pod_distrule_run.sh 9bc4db8bfcbd9bff6cd7c8f2be60135590c0ac96d9f77d4357b7115bc284b0b2
(LF endings checked by bytes in the tests; recompute the sha256 if any file is edited.) Pinned tool unchanged: 61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397.

## Other files
DISTRULE_DEFS.md (E1 to E7), test_dist_rule.py, smoke_cpu.sh.

## Scratch (all under distrule/tmp/, about 54 MB; nothing deleted, delete after reading)
tmp/pt, tmp/pt2 (pytest basetemp trees, synthetic worlds), tmp/synth0 (a synthetic world), tmp/smoke (8 CPU smoke outputs, `off`, `pinned`, `label`, `never`, `plumb`, `push`, `never_capo`, `label_capo`) and tmp/smoke_*.log.
