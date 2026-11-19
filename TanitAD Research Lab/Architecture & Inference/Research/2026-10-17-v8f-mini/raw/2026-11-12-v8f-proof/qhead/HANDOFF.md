# HANDOFF: QHEAD tool (card PREREG_V8F_PROOF_ADD42), builder state 2026-11-19

State: built and tested on the dev box. No git, no pod, nothing pushed. Folder `F:/TanitAD-scratch/v8proof/qhead/` (5 MB, `tmp/` 4 MB). Definitions and every REVIEW item: `QHEAD_DEFS.md`. Tiers: fit T0, rollouts T1-kin; no number here is a result.

## Files to push (to `/workspace/v8proof/distrule/`, next to the banked tools; compare sha256 on both sides)
| file | bytes | sha256 |
|---|---|---|
| qhead_model.py | 4,415 | f6ff755e3dd4e21247c9d4034a39deb26ac09c6b9a7135b05534278df639de0b |
| qhead_fit.py | 14,833 | 230aef333b397ef15d867f7ad7dbde4e5e4f227fcd0988a24b08ccc71ad9fb59 |
| v8f_t1_eval_rule_qhead.py | 7,461 | 21cf7ac38d344fd715eecc80c0322eedeca433a546955fa7542a397a5fbc8ae0 |
| pod_qhead_queue.sh | 10,415 | 480d3a759c761d3576fff97c10b25ce60b1c2550ba01e670a0f74e3004b1e272 |
| qhead_arms.txt | 318 | 5347ace6e65db5ed650093ef62399643e907ed7eea6fd67841483992057f52db |
Also `spatial/leadsup/read/train_pack.npz` (5,022,374 bytes, sha256 starts 9cfff95ffd) to `distrule/train_pack.npz` if the pod has no copy. Needed on the pod already: `v8f_t1_eval_rule_mix.py`, `v8f_t1_eval_rule.py`, `dist_rule.py`, `cmp_dumps.py`, `pod_ruleattr_run.sh`, the checkpoints `runs/leadsup_a|b/<ARM>/ckpt_last.pt`, the dump `runs/distrule/t1/LSRH_s0`.
The train states `<ARM>_train.npz` + `.json` for LS_s0, LS_s1, LSD_s0, LSD_s1 (156 MB each; optional `_train_lead.npz`, `_train_lead.npz.head.npz`) must sit in `QH_STATES` (default `/workspace/v8proof/qhead_states`; I do not know where the pod keeps them: set the variable).

## Pod commands
```
cd /workspace/v8proof/distrule && sha256sum qhead_*.py v8f_t1_eval_rule_qhead.py pod_qhead_queue.sh qhead_arms.txt
DRY=1 QH_STATES=<folder> bash pod_qhead_queue.sh          # prints 4 fits, the control rollout + cmp, 4 rollouts, a slot wait before each
QH_STATES=<folder> nohup setsid bash pod_qhead_queue.sh > /workspace/v8proof/qhead_start.out 2>&1 &
```
End markers in `/workspace/v8proof/qhead_run/`: `QHEAD_DONE.txt` (count of T1_DONE) or `QHEAD_STOP.txt` (reason). A second start is refused by `QHEAD_STARTED.txt`. The fits take about 3 minutes each on the CPU (MEASURED 5 s per 600 steps on the dev box, 19,164 steps).

## Pull list
Rollouts (`Q=distrule bash spatial/q3/pull_spatial_t1.sh LSQH_s0 LSQH_s1 LSDQH_s0 LSDQH_s1 LSQHI_s0`): about 25 MB each. Small files into `qhead/pulled/`: `qhead_run/heads/*.qhead.npz` (4 x 331 KB), `*.qhead.json` (4 x 3 KB), `qhead_run/qhead_queue.log`. Each rollout's `ticks/` carries `rule_log_<ARM>.npz` and `rule_log_<ARM>_qh.npz`. Then on the dev box: `bash qhead/qhead_read.sh` (readers r2, h3 per arm, names; last line `ZZ READ-QHEAD DONE`).

## Tested (CPU, synthetic, tiny; `pytest --basetemp qhead/tmp/pt -p no:cacheprovider`)
53 tests green: fit 16 (history, rows, pinball, schedule, gradient check h6 incl. regression, planted coverage 0.20 +- 0.03, conditional coverage by a state direction, h4 refusals, end to end), rule tool 11 (min of the two reads against an independent forward, float8 batch, last-5-frames, equals-head bit for bit, NaN, refusals, side file), h3 and names 17 (controls first, withheld outcomes, every branch), queue 9 (DRY order, markers, slot wait with a stub `nvidia-smi` and stub cgroup files, no pgrep/pkill/rm). `run_mutations.py`: 19 source-line mutations, 19 red, unmutated suite green (`MUTATIONS_QHEAD.json`, h6 evidence). Real data on this box: LS_s0 rows used 32,025 (equals the card), 26,327 fit / 5,698 held out; real gradient check passed (new head non-zero, control and the six banked tensors exactly 0.0); a 600-step fit ran (plumbing only, no coverage is quoted).

## Not tested
No rollout with the pinned tool and a model was run (h1 and the side-file alignment with the rule log are first seen on the pod). `qhead_h3.py` was not run on real eval files: it assumes state frame = cache row j and compares float16 eval states with the float32 rollout against 1e-4 m (REVIEW Q11). `qhead_read.sh` was not executed (no dumps); the JSON keys it reads come from the ADD31 outputs. The queue ran against stubs only; the sed that derives `pod_qhead_run.sh` was checked on the ADD31 argv line, check `grep -c -- '--head-p-min'` = 1 on the pod. The fit was not run for the full 19,164 steps.
