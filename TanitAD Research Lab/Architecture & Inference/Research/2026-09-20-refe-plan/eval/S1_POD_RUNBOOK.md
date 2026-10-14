# S1 pod runbook: RL-S1 finished -> u400 -> held-out data and labels -> nine scorer fine-tunes (aux drivable-area head) -> a private HF repo (2026-10-13)

Chain `refe/s1aux_pod.sh`, helper `refe/s1aux_checks.py`, status `refe/s1aux_status.sh`, tests `refe/test_s1aux_chain.py` (`ZZTEST_PASS 348`, MEASURED on the dev box, about 4 min).
What the arms decide: `eval/PREREG_AUX_DRIVABLE_S1.md` (IN FORCE, with Amendment A2). The argv per arm: `raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md` section 3 (the chain's `arm_args` equals it string for string; the test diffs it). The gate: profile `refe_aux_s1` of `refe/launch_gate.py`. The audit of the inputs: `raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md` (verdict (b): the A40 pod).
PI order on the A40: **P2 -> RL-S1 -> S1 -> P6.** Nothing here has run on a pod. Every cost below is ESTIMATED unless a source says MEASURED. Tier of everything S1 reads: **T0-like** (held-out selection and perception on teacher-forced banked sets), never driving performance.

## 0. Decisions the chain cannot make (Master Mind / PI) - read first

| id | decision | state and why it blocks |
|---|---|---|
| **S1-D1** | **Where S1 runs.** The PI is choosing; this chain is built for the recommended option, the A40 pod after RL-S1. | If the dev box is chosen instead, this chain does not apply (the audit's pixel relay of 21.6 GB per seed is then a prerequisite). |
| **S1-D2** | **How many train teacher-lane rows to regenerate.** `S1AUX_TLANE_N` (default **15,000**, SFT-4R's registered subset, seed 20261004). | The queue SFT-4 drew its lane subset from was never banked (PREREG_SFT4R, NAMED GAP), so a regenerated subset is a different draw of the same recipe. u400 itself trained on **2,109** rows. The chain REFUSES below 2,109. Cost ESTIMATED: 10.2 s per set (MEASURED for `rl_tlane_label.py` on the P2 pod) to 13.6 s (MEASURED on the dev box for this labeller) over about 10 workers = **4.3 to 5.7 h for 15,000**, 0.7 to 0.9 h for 2,500. The same seed gives a PREFIX of the same shuffled order, so `S1AUX_TLANE_N=2500` is a subset of the 15,000. The GPU idles during it. |
| **S1-D3** | The registered drop of `H1-s1` / `H1-s2` (PREREG_AUX_DRIVABLE_S1 section 1, proposal section 2): after `H1-s0` is read. | `S1AUX_ARMS="N-s0 H2-s0 H1-s0 N-s1 H2-s1 N-s2 H2-s2"` (a list must be a subsequence of the registered order, or the chain refuses). Fewer than three N runs caps every classification (section 4.3 of the prereg). |
| **S1-D4** | RL-S1 ended without `ZZRLS1_DONE` (a killed arm, a refusal). | The chain REFUSES (it does not wait forever for something that cannot happen). `S1AUX_RLS1_ABANDONED=1` only on the Master Mind's word. |
| **S1-D5** | **P6 is not aware of S1.** `refe/p6_pod.sh` checks only P2's lock and no GPU. | Start P6 by hand only after `ZZS1AUX_DONE`. Not edited here (outside scope); an integration item for the Master Mind (a lock check in `p6_pod.sh`). |
| S1-D6 | Not built: an in-chain K-rule monitor, an egress watcher, the P1/P2/P3 offline reader, any navtest read. | 400-update runs write ONE checkpoint (at update 400), so the SFT-4R watcher has nothing to watch; the arm is pushed when it ends (about 1.3 to 1.6 h at risk). K2/K3/K6 and P1/P3 are read from the arm evidence afterwards (`eval/aux_s1_readout.py`, `eval/aux_navtest_forward.py`). H2 is a held-out selection verdict only (prereg section 4.2). |

## 1. Before the pod (dev box)

1. **Land the package, then write the manifest** (the launch gate has `require_commit`): `python refe/launch_gate.py ship-manifest --pkg <package root> --commit <landed 40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json`. The committed `SHIP_MANIFEST.json` predates `scorer_finetune.py`'s head wiring and this chain, so **it must be regenerated** (the chain's precheck compares the shipped tree with it, file for file, and refuses a drift).
2. **Ship to a directory of its own, `/workspace/refe-plan-s1aux`** (NOT the RL-S1 or P6 package: their gates re-measure those trees). Ship `code/ refe/ splits/ SHIP_MANIFEST.json`, `raw/2026-10-04-lane-discipline/heldout_{navsim,teacher}_lane_labels.jsonl.xz` (the held-out lane labels SFT-4 trained against, 3,137 rows each) and `raw/2026-10-11-aux-drivable/head_proto_selftest.out` (the gate's suite reads it), keeping those two `raw/` paths. The chain's first lines list every required file.
3. **Ship the aux label store (352 MB, `store_sha256` `4a17db67...f287a`, 103,139 frames) to `/workspace/data/aux_drivable/label_store/`.** The chain verifies it by the store's OWN sha check (`AuxLabelSource` + every shard's sha256) and refuses otherwise. Suggested relay (the dev-box uplink is 1.2 MB/s MEASURED, so about 5 min): `python refe/hf_egress.py push --repo Sayood/tanitad-refe-s1aux --prefix inputs --tag s1_inputs --files <label_store.tar>` from the dev box (tar made with `tar -cf label_store.tar -C D:/Projects/TanitAD-artifacts/aux_drivable label_store`), then on the pod `huggingface_hub.hf_hub_download` and `tar --no-same-owner -xf ... -C /workspace/data/aux_drivable`.
4. `HF_TOKEN` (write-scoped) as a pod environment variable; never in a file or in chat. u400 is already in `Sayood/tanitad-refe-rls1` `base/model_sft4_B_u400.pt` (RLS1 runbook section 1.3); the chain fetches it, and verifies size + md5 `8e382eeb...` + sha256.
5. Pod facts the chain CHECKS and does not create: `/workspace/teacher_env.sh` (P2 / RL-S1 built it), `/workspace/data/backbones`, the P2 inputs (`refe_navtrain/train_grow`, `refe_navtrain/onpolicy/sets` carved, `navtrain_pixels` 32 shard dirs, `refe_sft2/train_pdm`), `navtrain_dbs` (1,192 DBs; fetched by `code/pod_dataprep.sh` when fewer).

## 2. Start (one command, detached) and dry runs

```
mkdir -p /workspace/data/s1aux && nohup setsid bash /workspace/refe-plan-s1aux/refe/s1aux_pod.sh \
  > /workspace/data/s1aux/stdout.log 2>&1 < /dev/null &
```
Start it while RL-S1 is **running or finished**. Since **A4 (2026-10-14)** the CPU-only stages (`env`, `fetch`, `data`, `relabel`, `bank_labels`) run **beside** RL-S1's training (labellers at `nice -n 19`, cgroup cores minus 4 workers). Stage `gpu_wait`, before the first arm, then WAITS (log line `ZZS1AUX_WAITING <why>` every 10 min) until the P2 and RL-S1 locks are free, RL-S1 wrote `ZZRLS1_DONE`, and the GPU has been idle on 3 consecutive polls. It re-checks the GPU right before every arm and never puts a GPU job beside a training job. Guard: if RL-S1's update rate (its egress ticks) drops by more than 5 % during the overlap, stop the labellers by explicit PID (`PREREG_AUX_DRIVABLE_S1.md` A4).
`S1AUX_STOP_AFTER=<precheck|env|fetch|data|relabel|bank_labels|gpu_wait|gate_N-s0|arm_N-s0|...>` ends the chain after that stage. Cheap dry runs, in this order: `precheck` (seconds; no wait since A4), `fetch` (u400, minutes), `data` (held-out bank, lane labels, the store check, minutes), `relabel` (hours: S1-D2), `bank_labels`, then `gate_N-s0` (one preflight, one token: a G-SUITE run, minutes).
Knobs: `S1AUX_ARMS`, `S1AUX_TLANE_N`, `S1AUX_RLS1_ABANDONED`, `S1AUX_WAIT_MAX_S`, `S1AUX_POLL_S` (30), `S1AUX_IDLE_POLLS` (3), `S1AUX_DD_MB` (24576), `S1AUX_LABEL_WORKERS`, `S1AUX_TICK` (1), `LANE_QUEUE_SRC` (a directory holding `props_*.jsonl*`; default: the HF banks' `onpolicy_queue.tar.zst`, md5 from the repo's `MD5SUMS.txt`).
Refusals are `ZZS1AUX_REFUSED stage=<stage>: <why>` in `/workspace/data/s1aux/chain.log` (and on HF: the exit trap pushes it).

## 3. What the chain does, in order (idempotent: inputs are re-verified by content on every start; expensive stages leave `stage/<name>.ok`)

| stage | what | marker |
|---|---|---|
| precheck | `HF_TOKEN`; every shipped file; `teacher_env.sh`; **the shipped tree equals `SHIP_MANIFEST.json`**; the arm list; **WAIT** for P2 / RL-S1 / the GPU; a WRITE test of `S1AUX_DD_MB` (never `df`) | `ZZS1AUX_PRECHECK_OK` |
| env | backbones dir, cgroup cores, label workers, the 30-min `chain.log` push | `ZZS1AUX_ENV_OK` |
| fetch | u400 from `Sayood/tanitad-refe-rls1`; size + md5 + sha256 against the pin; **a wrong file is deleted** (the argv's `--ckpt` points at that path) | `ZZS1AUX_FETCH_OK` |
| data | the P2 inputs exist; the held-out bank (7 files, 24 logs) from the pod's tarball or HF (md5 `1f43409b...`), `safe_extract` (no chown, no `..`); held-out lane + teacher-lane labels restored from the repo's `.xz` (3,137 rows each, refused otherwise); the aux store verifies | `ZZS1AUX_DATA_OK` |
| relabel | the labellers can build (`PT._setup()`); **`heldout_pdm`** by `onpolicy_relabel_pdm.py --served-only` (p2r_pod's recipe) to `pdm_label_complete.py --expect 3137` and >= 3,137 rows; **`train_teacher_lane`**: `select_bank_subset.py` (seed 20261004, `--n S1AUX_TLANE_N`) from the queue, then both ranks of `onpolicy_relabel_teacher_lane.py --repair-last-heading`, each ending in `ZZTEACHER_LANE_DONE`, >= 2,109 rows | `ZZS1AUX_RELABEL_OK` |
| bank_labels | tars of `heldout_pdm`, `train_teacher_lane`, both held-out lane dirs + sha256 manifest pushed to `Sayood/tanitad-refe-s1aux` `labels/`, **proven by the remote listing and content** | `ZZS1AUX_LABELS_BANKED` |
| arms | per arm, below | `ZZS1AUX_ARM_*` |

## 4. Per arm (registered order `N-s0 H2-s0 H1-s0 N-s1 H2-s1 N-s2 H2-s2 H1-s1 H1-s2`)

1. **Restart rule.** If a previous attempt STARTED training (`stage/trainstart_<ARM>.ok`) and did not finish, its directory is renamed `runs/<ARM>_killed_<k>` (its token dir `lg/<ARM>_killed_<k>`), a `restart` row goes to `arms_manifest.jsonl`, its logs are pushed best-effort, and a fresh preflight and a **fresh token** follow. Never stitched, never resumed (`scorer_finetune.py` has no resume).
2. `scorer_finetune.py <argv> --preflight` -> `ZZSFT_PREFLIGHT_PASS` (G1-G11; G12 on a head arm) -> `s1aux_checks.py data-check` on the preflight's `data` event: **train_sets 174,912, heldout_sets 3,137, 24 logs, 0 train rows in a held-out log, teacher-lane >= 2,109, PDM loaded** (the SFT-4 defect class: a named dir that loads nothing).
3. `launch_gate.py run --profile refe_aux_s1 --arm-name <ARM>` -> a PASS token -> `token_ok` (the gate's `verify`, which RE-MEASURES argv, tree and data and checks the HMAC; never a grep). The token id goes to `arms_manifest.jsonl` BEFORE the first step (prereg section 2).
4. `wait_clear` (the GPU again), `token_ok` again, `trainstart` mark, `scorer_finetune.py <argv>` in a subshell without `HF_TOKEN` and without fd 200. **The trainer is never started without a verified token.**
5. `ZZSFT_DONE` + `s1aux_checks.py run-check`: `start` and `done` read **updates == 400**, evals at `step0` and `update400`, `dump/scores_*` (every arm, N included), a head arm has `aux_drv.pt`, `aux_eval` rows and `dump/head_pred_*` for both tags and finite `aux_bce`, **an N arm has none of them**, `model_sft_B.pt` >= 1.2 GB. A run that fails this is not an S1 arm and is NOT banked.
6. Egress to `Sayood/tanitad-refe-s1aux` `arms/<ARM>/`: `model_sft_B.pt`, `aux_drv.pt`, `scorer_states_last.pt`, logs, preflight, `PREFLIGHT_PASS.json`, the PASS token, `run_check.json`, `arms_manifest.jsonl`, `dump/*.npz`. Success = `hf_egress.py`'s listing size check AND `s1aux_checks.py remote-verify` (LFS sha256, or the git blob id of a small file) of every file. `model_sft_A.pt` (not an S1 arm) stays on the pod.

## 5. Timeline (ESTIMATED; one A40, 11 vCPU)

| step | estimate | source |
|---|---|---|
| fetch u400 + held-out bank + queue (3.1 GB) | minutes (pod downlink UNMEASURED) | - |
| `heldout_pdm` | 3 to 20 min (3,137 sets x 0.357 s / 1 to 10 workers) | audit section 3, the pod's relabel log rate |
| `train_teacher_lane` | 4.3 to 5.7 h at 15,000 sets (S1-D2) | see S1-D2 |
| one arm | 1.3 to 1.6 h (400 updates x 11.4 to 14.4 s) + preflight / gate | proposal section 2 |
| nine arms | 11.7 to 14.4 h | - |
| **whole chain** | about **16 to 21 h**; about 12 to 16 h with `S1AUX_TLANE_N=2500` | - |

## 6. Watch with a script, and the markers

- **On the pod** (read-only, starts nothing): `bash /workspace/refe-plan-s1aux/refe/s1aux_status.sh` prints `ZZS1AUX_STATUS <running|waiting|done|refused|dead|none> arms_done=... now=<arm>:<step> stages_ok=<n> last=<last marker line>` and exits 0 / 9 / 10 / 11 / 12 / 13. A wait loop for a Monitor or `until`: `until bash .../s1aux_status.sh > /tmp/s1.out; [ $? -ge 10 ]; do sleep 120; done; cat /tmp/s1.out` (it stops on done, refused, dead, none; the markers are anchored on the timestamp, so the loop's own command line cannot match).
- **Without the pod:** HF (private) `Sayood/tanitad-refe-s1aux`: `logs/chain.log` every 30 min and at exit, `labels/` once, `arms/<ARM>/` when an arm ends, `arms/<ARM>_killed_<k>/` for a restarted attempt.
- **`chain.log` markers, in order:** `ZZS1AUX_START`, `ZZS1AUX_WAITING <why>` (0 or more), `ZZS1AUX_PRECHECK_OK`, `ZZS1AUX_ENV_OK`, `ZZS1AUX_FETCH_OK`, `ZZS1AUX_DATA_OK`, `ZZS1AUX_RELABEL_OK`, `ZZS1AUX_LABELS_BANKED`, per arm `ZZS1AUX_ARM_GATE_OK`, `ZZS1AUX_ARM_TRAIN_START`, `ZZS1AUX_ARM_TRAINED`, `ZZS1AUX_ARM_DONE <ARM>` (`ZZS1AUX_ARM_RESTART <ARM> k=<k>` before them after a kill), finally `ZZS1AUX_DONE arms: ...`.
- `ZZS1AUX_REFUSED stage=<precheck|env|fetch|data|relabel|bank_labels|arm_<ARM>>: <why>` = stopped on purpose (exit 1). `ZZS1AUX_WARN` = a best-effort push of a killed attempt's logs failed. `ZZEGRESS_FAIL` = a push failed; **the evidence is on the pod only: do not delete the pod.**

## 7. Restart, exactly (no addresses)

- **Any interruption, same pod:** run the section-2 command again. Finished stages are skipped; inputs are re-verified; the labellers resume by key; an arm interrupted in training is archived as `<ARM>_killed_<k>` and restarts at update 0 under a fresh token (at most about 1.6 h lost).
- **After `ZZS1AUX_REFUSED`:** read the line, fix the named cause, run the command again. A refusal at `precheck` about the tree means the manifest is stale: re-run section 1.1 and re-ship.
- **A trainer that exits non-zero:** the refusal names the tail of `sft.log`; re-running restarts that arm (above). A deterministic failure (OOM at `--batch 8` with the head: UNVERIFIED on the A40) repeats; read `preflight.jsonl`'s `gpu_mem_gb` and G12 first. K3 (> 1 GB retained head activations) or an OOM STOPS the programme and is reported, not worked around (proposal section 2).
- **A NEW pod after the old one was lost:** ship section 1, set `HF_TOKEN`, run the command. The chain re-derives everything it needs except the banked labels: fetch `labels/s1_*.tar` and `MANIFEST_s1_labels_sha256.txt` from `Sayood/tanitad-refe-s1aux`, check each sha256, `tar --no-same-owner -xf ... -C /workspace/data/refe_sft2`, and the relabel stage will find the labels and only verify them (UNTESTED on a pod).

## 8. Evidence status of this build

- **MEASURED (dev box):** `bash -n`; `test_s1aux_chain.py` `ZZTEST_PASS 348`: the nine argv equal the proposal's section 3 string for string and the REAL launch gate judges all nine CLEAN and eight mutated argv REFUSED; `token_ok` against tokens minted by the gate's own `mint()` for `refe_aux_s1`; the REAL chain run in a sandbox (stub trainer / gate / egress / labellers; the REAL helper and the REAL aux label store) through every stage's refusal path, a two-arm happy path in order, no trainer start without a verified token, a failing arm never letting the next run, and a kill-restart; 34 static mutations of the chain each turn `check_chain` RED.
- **UNVERIFIED (never run on a pod, not probed read-only):** the held-out tarball's internal layout (`refe_heldout/{sets,heldout_logs.txt}`) and that the pod's copy sits at `/workspace/data/hf_dl/heldout/`; that `onpolicy_queue.tar.zst` holds `props_r{0,1}_*.jsonl` files in exactly one directory; that a regenerated lane subset reaches 2,109 rows; the runtime of the two relabellers on 11 vCPU; `REFE_SIM_HZ` (left at the module default 20, as `p2r_pod.sh` did for `train_pdm` and `sft4r_relabel.sh` does; `rls1_pod.sh` sets 10 for its own labeller: whether either rate matters for these two labellers was not established); `zstd` on the pod; that the gate's G-SUITE (five scripts, including `test_aux_drivable_head.py` with `AUX_LABEL_STORE` set to the pod's store) passes on the pod; the GPU fit of ViT-L at `--batch 8` with the head resident; the audited data counts (174,912 / 3,137 / 24) on the pod's real `data` event; the pod's disk quota against `S1AUX_DD_MB`.
