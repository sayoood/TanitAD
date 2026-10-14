# P2-s1 pod runbook: the P2 training-seed replicate, after RL-S1 and S1 (2026-10-14)

Chain `refe/p2rep_pod.sh`, read-only helper `refe/p2rep_checks.py`, launch-gate profile `refe_p2rep` (`refe/launch_gate.py`), tests `refe/test_p2rep_chain.py` (`ZZTEST_PASS 350`, MEASURED on the dev box, about 3 min; section 9).
What the arm decides: `eval/PREREG_P2_REPLICATE.md` (the Master Mind's, written in parallel; this runbook does not read it). The recipe: `eval/PREREG_P2.md` + `eval/PREREG_P2_AMENDMENT_1.md`. PI ruling 2026-10-14: **"Replicate p2"**.
PI order on the A40: **P2 -> RL-S1 -> S1 -> P2-s1.** Nothing here has run on a pod. Every cost below is ESTIMATED unless a source says MEASURED. Tier of the arm's own numbers: T0-like until a navtest read (T1 = NAVSIM PDMS, self-action open loop), never driving performance.

## 0. Integration items for the Master Mind (headline) and decisions the chain cannot make

| id | item | state |
|---|---|---|
| **P2REP-I1** | **Land, then regenerate `SHIP_MANIFEST.json`.** The committed manifest predates `p2rep_pod.sh`, `p2rep_checks.py`, `test_p2rep_chain.py` and the `launch_gate.py` change. The chain's precheck compares the shipped tree with the manifest file for file and REFUSES a drift. | section 1.1 |
| **P2REP-I2** | **`launch_gate.py` changed (shared by every profile).** Added: the `refe_p2rep` profile and one generic rule, `pinned_argv` (a profile that carries it accepts ONE argv; no other profile carries it). `test_launch_gate.py --static --no-banked` and the three chain tests are unchanged in count (section 9). The RL-S1 and S1 packages already shipped hold their own copies and are untouched; if either is re-shipped from a commit that includes this change, regenerate ITS manifest too. | MEASURED |
| **P2REP-I3** | **Start order.** Start this chain only after S1's chain has been started (S1 holds `chain.lock` from its first line). S1 not running and no `ZZS1AUX_DONE` in `/workspace/data/s1aux/chain.log` is a REFUSAL, not a wait. `P2REP_S1_ABANDONED=1` only on the Master Mind's word. | by design (a chain that waits for something that cannot happen is worse than one that refuses) |
| **P2REP-I4** | **P6 does not know about this chain.** `refe/p6_pod.sh` checks only P2's lock and no GPU (S1-D5). This chain WAITS for an idle GPU, so a P6 that is already running delays it, but a P6 started by hand while this chain trains would add load to a training pod. Start P6 only after `ZZP2REP_DONE` unless the Master Mind reorders. Not edited here (outside scope). | integration item |
| P2REP-D1 | `P2REP_P2_PROVENANCE_WAIVED=1` waives ONLY P2's own log markers (`ZZP2R_DONE` in P2's `chain.log`, `weights.log`, `carve.log`, `pix.log`). Every content check stays. | needed only if the pod's P2 logs differ from what `p2r_pod.sh` writes (UNVERIFIED, section 9) |
| P2REP-D2 | The registered served count (174,912) is a constant of the chain and of the helper, not an input. There is no override. | parity |

## 1. Before the pod (dev box)

1. **Land the package, then write the manifest** (the launch gate has `require_commit`): `python refe/launch_gate.py ship-manifest --pkg <package root> --commit <landed 40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json` (it refuses any file that is not its blob at that commit).
   The bytes that ship must be the landed ones, LF-terminated: materialise the package from the mirror with **`git -c core.autocrlf=false archive`** (a checkout with `core.autocrlf=true` turns LF files CRLF and every sha256 in the manifest then differs; the precheck refuses it by name). The mirror, not the D: worktree, holds the landing (the worktree lags).
2. **Ship to a directory of its own, `/workspace/refe-plan-p2rep`** (NOT `/workspace/refe-plan`, `refe-plan-rls1`, `refe-plan-s1aux` or P6's: each chain's launch gate re-measures its own tree, and adding files there would break it). Ship `code/ refe/ splits/ SHIP_MANIFEST.json` (no `raw/`, no `eval/`). The chain's first lines list every required file.
3. **`HF_TOKEN`** (write-scoped for `Sayood`) as a pod environment variable, never in a file or in chat. The chain creates `Sayood/tanitad-refe-p2-joint-pdm-s1` (PRIVATE) with its first push. **It never writes to `Sayood/tanitad-refe-p2-joint-pdm`**: every `p2r_sync.py` call in the chain names `--repo "$OUT_REPO"` (the test mutates each one away and requires RED, and the sandbox run records every call's repo).
4. Pod facts the chain CHECKS and does not create: `/workspace/teacher_env.sh` (P2 built it), the P2 inputs under `/workspace/data` (section 3), the locks and logs of RL-S1 (`rls1/`) and S1 (`s1aux/`). Nothing is built, carved, relabelled or downloaded.

## 2. Start (one command, detached) and dry runs

```
mkdir -p /workspace/data/refe_p2rep && nohup setsid bash /workspace/refe-plan-p2rep/refe/p2rep_pod.sh \
  > /workspace/data/refe_p2rep/stdout.log 2>&1 < /dev/null &
```
`HF_TOKEN` comes from the environment of that shell. The chain WAITS (log line `ZZP2REP_WAITING <why>` every 10 min) until the P2, RL-S1 and S1 locks are free, S1 logged `ZZS1AUX_DONE`, and the GPU was idle on 3 consecutive polls; it re-checks the GPU before the preflight and before every `train.py` launch. It never starts beside a training job.
`P2REP_STOP_AFTER=<precheck|wait|verify|recover|gate|launch_gate>` ends the chain after that stage (a dry run, in this order): `precheck` (seconds), `wait`, `verify` (the shared inputs, minutes), `recover`, `gate` (p2_gate G1-G4), `launch_gate` (the PASS token, ESTIMATED 15-25 min). A dry run through `launch_gate` leaves the stage markers, so the real start skips them and only re-verifies the token.
Knobs: `P2REP_S1_ABANDONED`, `P2REP_P2_PROVENANCE_WAIVED`, `P2REP_WAIT_MAX_S` (default unbounded), `P2REP_POLL_S` (30), `P2REP_IDLE_POLLS` (3), `P2REP_DD_MB` (8192), `P2REP_SYNC_MIN` (30), `P2REP_HEAVY_MIN` (30), `P2REP_NVSMI`.

## 3. What the chain does, in order, and what it DROPPED from `p2r_pod.sh`

`p2r_pod.sh` (P2's chain, UNTOUCHED) built the inputs; this chain only verifies them. Stage by stage:

| `p2r_pod.sh` stage | here | why |
|---|---|---|
| precheck (HF_TOKEN, files) | KEPT, plus `p2rep_checks.py tree` (shipped tree == manifest) | a mistake costs seconds, before any wait |
| env (`pod_teacher_env.sh`) | **verify-only**: `teacher_env.sh` must exist | an env build beside a training job |
| weights (`pod_train.sh`, downloads) | **verify-only**: sha256 of `dinov3-vitl16/model.safetensors` against `W_SHA[vitl16]` parsed from the shipped `code/pod_train.sh` | no download into `backbones/` |
| restore (HF: init checkpoint + two banks) | **DROPPED** | it downloads and extracts into shared dirs. The init checkpoint is verified (md5 + bytes from `p2r_sync.PINS`), the banks by P2's own definitions (next rows) |
| recover (HF: labels + ckpt of a lost pod) | **CONVERTED**: this arm's `ckpt_last.pt` only, from this arm's repo, into a scratch work dir (`refe_p2rep/recover_work`) | the PDM-label branch of `recover` cannot reach `refe_sft2/` |
| carve (`p2r_carve.py`: MOVES files in `onpolicy/sets`) | **DROPPED**; its invariants are checked read-only | RL-S1 and S1 read `sets/`. Served count by `onpolicy_relabel_pdm.served_keys` == 174,912 and ZERO rows of the 24 held-out logs (the carve's own `log_name_of`, every row) |
| dbs, relabel (PDM labelling) | **DROPPED** | the labels exist. P2's own `pdm_label_complete.py --expect 174912` re-audits them read-only (it writes only its `--json`, in this arm's dir). The navtrain DB count (`dbs=` in the log line) is informational: `train.py` never reads them |
| pix | **verify-only**: `.done_1..32` (what `ZZOK pix 32/32 shards ZZ` counts) and 32 shard dirs | |
| pdm_push (HF) | **DROPPED** | P2 banked the labels; this chain never pushes them |
| gate (`p2r_gate_prep.py` + `p2_gate.py`, G1-G4) | KEPT, in `refe_p2rep/{gate_sets,gate_pdm,gate}` | G-P2GATE is a required sub-check of the profile |
| launch_gate | KEPT with **profile `refe_p2rep`** (section 4) | |
| preflight, train (supervised, 6 attempts) | KEPT; token VERIFIED before the preflight and before EVERY launch; GPU re-checked before each | `--resume` continues from `ckpt_last.pt` |
| final push (remote-verified) | KEPT, to this arm's repo | `ZZP2REP_FINAL_OK` carries `model_final`'s sha256 |

Stage order in `chain.log`: `ZZP2REP_START`, `ZZP2REP_PRECHECK_OK`, [watcher starts] `ZZP2REP_WAIT_OK`, `ZZP2REP_INPUTS_OK served=174912 ...`, `dd write test`, `ZZP2REP_RECOVER`, `ZZP2REP_GATE_OK`, `ZZP2REP_LAUNCH_GATE_OK`, `ZZP2REP_TRAIN_START`, `attempt N exited rc`, `ZZP2REP_TRAIN_DONE`, `ZZP2REP_FINAL_OK model_final sha256 <hex> remote=sha256-verified`, **`ZZP2REP_DONE`**.
Everything is written under `/workspace/data/refe_p2rep/` (stage markers, logs, `lg/`, gate dirs) and the run dir `/workspace/data/refe_runs/p2_joint_pdm_s1`. A static scan of the chain (and the sandbox run: the shared tree is byte- and mtime-identical before and after) shows no other write target.

## 4. The arm and the gate decision

- **The argv** is `ARGS` of `p2r_pod.sh` token for token with exactly two changes: `--out /workspace/data/refe_runs/p2_joint_pdm_s1` and `--seed 1` added (before `--out`). Same init checkpoint, same data paths, same everything. P2 ran the parser default (seed 0).
- **Why a new profile.** `refe_p2` pins 9 of the argv's 24 flags and neither `--seed` nor `--out`: MEASURED, `profile_refusals(refe_p2, ...)` is CLEAN on the replicate's argv, on the same argv with `--seed 1` dropped, with `--seed 7`, and with P2's own `--out`. The existing G-DVB seed check (`gate_checks.py`: argv seed == `torch.initial_seed()` of the BUILD run) would catch a trainer that ignores the flag, but nothing makes the argv carry `--seed 1`. `refe_p2rep` = `refe_p2` (same checks, same G-P2GATE, same data and forbidden flags) with `--seed 1` required and the WHOLE argv pinned, token for token, order-blind.
- **Deliberate-regression arms** (all in `test_p2rep_chain.py`, 23 gate mutations): seed dropped, 0, 2; P2's own `--out`; another `--out`; each of lr / epochs / accum / batch / init / pdm labels / workers / ckpt cadence / log-every moved; `--tf32`, `--compile`, `--resume` dropped; a lever added; `--seed` doubled; a forbidden flag; a stray leading token. Each is REFUSED by the REAL `profile_refusals`; `run_checks` marks every check FAIL; `finalize` mints NO token; `verify_token` refuses the result.
- **Reachability of `--seed 1`** (`train.py`): parser `--seed` int, default 0 (line 1920) -> `a.seed` -> `torch.manual_seed(a.seed)` (line 1125, before `REFe(cfg)`, line 1126, so the model's fresh modules draw from it) -> `SceneEpochSampler(ds.scene_groups, a.seed, ...)` (1262; `EpochSampler` 1264) and the eval loader's samplers (1500, 1501); `SceneEpochSampler` derives its per-scene offsets and every epoch's order from `seed` (lines 937, 946). The resume identity (`IDENTITY_ARGS`, line 965) names `seed`: a resume under another seed refuses. Checked by an AST pass over the real file, by a dynamic probe (`train.run()` with the model build stubbed records the seed it was given: 1 with the argv, 0 without) and by five source mutations that each turn the pass RED.
  Not routed through `a.seed` by this file: the DataLoader workers' base seed comes from torch's global generator (which line 1125 seeds; torch's default behaviour, not probed here), and `torch.Generator().manual_seed(i)` at line 790 is the `--synthetic` path (forbidden in a launch).

## 5. Estimated timeline (one A40, 11 vCPU; ESTIMATED, nothing here has run on a pod)

| step | estimate | source |
|---|---|---|
| wait for RL-S1 and S1 | their remaining time | S1 runbook section 5: about 16 to 21 h; RL-S1 runbook |
| precheck + verification (md5 of the 1.29 GB checkpoint, sha256 of the ViT-L weights, two line passes over the carved bank, the PDM audit) | minutes, UNVERIFIED | no pod timing banked |
| dd write test (8 GB) | seconds (1.2 to 1.3 GB/s MEASURED on this pod, `P2R_POD_RUNBOOK.md` section 6) | |
| p2_gate + launch gate | 15 to 25 min | `P2R_POD_RUNBOOK.md` section 2, UNVERIFIED |
| training: 1,209 steps at about 60 s, **about 20 to 21 h = 42 half-hourly egress ticks** | P2's own run (`RESULT_P2.md`: 1,209 steps) | |
| final push | under 15 min | `P2R_POD_RUNBOOK.md` section 5 |
| **P2-s1 itself** | **about 21 to 22 h** after the wait | |

## 6. Watch without SSH: HF private repo `Sayood/tanitad-refe-p2-joint-pdm-s1`

| file | updated | tells you |
|---|---|---|
| `logs/chain.log` (+ the other `*.log` / `*.json` of the work dir: `inputs_verified.json`, `pdm_complete.json`, `token_verify.json`, `launch_gate_PASS_token.json`) | every 30 min from the start of the wait | the markers above; `ZZP2REP_WAITING` while upstream runs |
| `run/metrics.jsonl`, `run/summary.json`, `run/config.json` | every 30 min once training writes them | training; `done: true` at the end |
| `run/ckpt_last.pt`, `run/snap_epoch*.pt` | every 30 min if changed | a pod loss costs the 20-min checkpoint interval + this cadence |
| `run/model_final.pt` | at the end | the result, sha256-checked against `summary.json` |

A wait loop for a Monitor (anchored on the timestamp so its own command line cannot match): `until grep -Eq '^[0-9T:-]+ ZZP2REP_(DONE|REFUSED|FAILED|FINAL_FAILED)' /workspace/data/refe_p2rep/chain.log; do sleep 120; done`. A `chain.log` on HF older than about an hour means the pod or the chain is down.
Failure markers: `ZZP2REP_REFUSED stage=<precheck|wait|verify|recover|gate|launch_gate|train>: <why>` = stopped on purpose (exit 1); `ZZP2REP_FAILED` = six attempts, no `done: true`; `ZZP2REP_FINAL_FAILED` = **the checkpoint is on the pod only: do not delete the pod**; `ZZSYNC_DEGRADED` = pushes are failing (token or quota), training continues.

## 7. Restart, exactly (no addresses)

- **Any interruption, same pod:** run the section-2 command again. Finished stages are skipped (`stage/gate.ok`, `stage/launch_gate.ok`), the inputs are re-verified by content, the token is re-verified against the exact argv, and `train.py --resume` continues from `ckpt_last.pt`.
- **After `ZZP2REP_REFUSED`:** read the line, fix the named cause, run the command again. `stage=precheck`, "not its SHIP_MANIFEST.json": the shipped tree drifted or the manifest is stale; re-run section 1.1 and re-ship. `stage=wait`, "S1 is neither running nor finished": start S1 or get the Master Mind's word for `P2REP_S1_ABANDONED=1`. **`stage=verify`: a shared input is not what P2 built (or S1 / RL-S1 changed it). Do NOT repair it from this chain and do NOT re-run a P2 stage; report it: other chains read the same files.** `stage=launch_gate`: read `lg/evidence_*.json`, fix the cause, delete `stage/launch_gate.ok`, re-run. A re-ship after the token was minted changes the tree: the same deletion applies.
- **A trainer that exits non-zero:** the chain re-checks the GPU and the token and relaunches (six attempts, 60 s apart); `train.log` has the cause.
- **A NEW pod after the old one was lost:** the shared inputs are P2's; rebuild them with P2's own chain first (`P2R_POD_RUNBOOK.md`). This chain's `recover` stage then brings back `ckpt_last.pt` from this arm's repo.

## 8. What this chain does not claim

It replicates the TRAINING seed only. The inference seed and the data order of the held-out reads are not varied here; the replicate's sampler draws a different epoch order and a different per-scene member offset because `--seed` changed, which is the intended variance, and the fresh modules of the model initialise from a different RNG state. Whether that is enough to read P2's R2 interval against a run-to-run floor is `PREREG_P2_REPLICATE.md`'s question, not this chain's.

## 9. Evidence status of this build

- **MEASURED (dev box, Git Bash, driverl-eval venv):** `bash -n`; the test's `ZZTEST_PASS` count and runtime in the hand-back; the three copies of the argv equal up to the two changes; the 23 gate mutations and 49 chain mutations each RED; the seed hops (AST, dynamic probe, five source mutations); `upstream` / `wait_clear` extracted verbatim and run under bash with lock and GPU shims (S1's lock held => waits, S1 free and no `ZZS1AUX_DONE` => refuses, `P2REP_S1_ABANDONED=1` => proceeds); `token_ok` against tokens minted by the gate's own `mint()`; the REAL chain in a sandbox (stub trainer / gate / egress, the REAL helper and the REAL `pdm_label_complete.py`) through every refusal, the happy path in order, no trainer without a verified token, every HF call on this arm's repo and none on P2's, `HF_TOKEN` only in the egress children, the shared tree unchanged; `test_flock_rw.py` over the new script.
- **UNVERIFIED (never run on a pod, not probed read-only; the Master Mind ships and launches):** that the pod's P2 logs are where `p2r_pod.sh` writes them (`weights.log`, `carve.log`, `pix.log`, `chain.log` with `ZZP2R_DONE` under `/workspace/data/refe_p2/`; waivable, section 0); that the carved bank still serves 174,912 and holds no held-out row after RL-S1 and S1 (the verification decides); the runtime of the two line passes over the bank and of the PDM audit on the pod; `/workspace` free space for this run's checkpoints (the 8 GB write test is a floor, not a measurement; `ckpt_last.pt`'s size is unmeasured here); the G-SUITE run of `test_p2rep_chain.py` on the pod (it prints no `ZZTEST_SKIP`, needs `bash`, `readlink /proc/self/fd`, and `torch` to import `train.py`); that `train.py --compile` and the gate's GPU stages behave as on P2's pod after RL-S1 and S1 have run beside them (leftover `/tmp/torchinductor_*` or GPU memory); that no process outside the three chains holds the GPU at the moment of the wait.
