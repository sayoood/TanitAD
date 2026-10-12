# RLS1 pod runbook: P2 finished -> frozen-fan dump -> labels -> five scorer-only arms -> a private HF repo (2026-10-12)

Chain `refe/rls1_pod.sh`, helpers `refe/rls1_checks.py`, tests `refe/test_rls1_chain.py` (`ZZTEST_PASS 177`; `refe/test_p2r_chain.py` is unchanged at 139).
What the arms decide: `eval/PREREG_RL_S1.md` (DRAFT, becomes binding only when landed before the first training step). The argv per arm: `eval/HANDOFF_RL_S1.md`, section "The exact argv per arm".
PI order on the A40: **P2 -> RL-S1 -> P6.** Nothing here has run on a pod. Every cost below is ESTIMATED unless a source says MEASURED.

## 0. Decisions the chain cannot make (Master Mind / PI) - read first

| id | decision | state and why it blocks |
|---|---|---|
| **TL-1** | **Where do the teacher-lane labels of the RL scenes come from?** | **BLOCKS THE CHAIN at stage `tlane`, before any compute.** u400 is a 7-output checkpoint; `scorer_finetune.py` REFUSES it without `--lane-head`, and `--lane-head` needs `--teacher-lane-train/-heldout` side files that reach the batch (its G10: `n_tlane > 0` on both sides and a soft target). `onpolicy_relabel_teacher_lane.py` reads the teacher simulator's QUEUE rows (proposals + teacher rollout), which the lean frozen-fan dump does not carry; the teacher simulator costs 10.2 s per set (HANDOFF part-2 finding F3). The pipeline named for this build (dump, PDM, rl_terms) produces no `teacher_lane_r*.jsonl`. The chain asks for them (`rls1_checks.py tlane-check`: key in the partition side, `ckpt_step` 10075, 64 finite `center_line.CenterLine.info`) in `$RLS1_TLANE_TRAIN` / `$RLS1_TLANE_HELDOUT` (default `/workspace/data/rls1/tlane_train`, `.../tlane_heldout`). Options: (a) label a SUBSET with the old queue path (SFT-4R's precedent was 2,109 train sets; 10.2 s per set, about 6 CPU-h for 2,109; a held-out subset too), the rest are masked by the trainer; set `RLS1_TLANE_MIN_TRAIN/_HELDOUT`; (b) a lean teacher-lane labeller (not built; it would be a different label from the teacher's `center_line` leaf unless it reuses `onpolicy_label.Scorer`); (c) a PI ruling to train without the lane output, which needs a `scorer_finetune.py` + gate-rule change (not mine). The chain's default floor is the trainer's own bar (one usable row per side). |
| **D-RLS1-1** | **Separate C0/C1 runs, or the paired design?** | Default `RLS1_ARMS="C0 R0 C1 R1 R2"` (5 runs, about 22 h of training). `RLS1_ARMS="R0 R1 R2"` (3 runs, about 13 h): arm A of an `--epo` run is BCE-only on the same sets, order, steps and trunk forward and `test_rl_epo.py` proves EPO never touches it, so it IS the matched control (the prereg calls this the cheaper design). K4 then reads R against that arm A (the chain falls back to it automatically). Each token is still minted per named arm. |
| D-RLS1-2 | K3's absolute g3 bounds (pick/fan-mean path >= 0.894, shortest-bin share <= 0.55) are the review's, derived on navtest. If the unchanged base already breaches them on RL-val, K3 would kill every R arm for a property it started with. The monitor reads `step0` first: if the base breaches, **K3 is switched off for that run and says so** (`ZZRLS1_K3_DISABLED_BASE_ALREADY_BREACHES_G3_ON_RL_VAL`); the final read still reports g3. K3/K4/K5 are applied to R arms only (a control is not judged against itself); C arms get K2 (non-finite loss) and K6 (6 h). | my design call, one `if` in `rls1_checks.monitor` |
| D-RLS1-3 | **K2's "grad norm > 20x the control's median" is NOT machine-checked**: the trainer does not log its clipped grad norm. K2 here is non-finite loss only. | needs a one-line log field in `scorer_finetune.py` (its owner) |
| D-RLS1-4 | Any pre-registered kill STOPS THE CHAIN after the arm's evidence is banked (marker `stage/arm_<ARM>_killed`); the Master Mind decides before another arm runs. A kill the live monitor missed is found by `replay` after the run and treated the same. | my design call |

### Master Mind rulings (2026-10-12, before any RL data)

* **D-RLS1-1: the PAIRED design, `RLS1_ARMS="R0 R1 R2"`.** The prereg already named it the cheaper option with within-seed pairing for free.
  It also makes C and R set-identical by construction, which closes the gate build's finding (a separate C run without `--epo` trains on every
  set). R2's arm A (seed 0) is a second training of R0's arm A: a free run-to-run noise floor (`H-ESTIM-SEED-1`). It saves about 9 h of A40.
* **D-RLS1-2: accepted.** K3 is switched off for a run ONLY when the unchanged base already breaches g3 on RL-val; the base's own values are logged next to it.
* **D-RLS1-3: accepted, with its limit stated.** K2 is non-finite loss only until the trainer logs its clipped grad norm. Adding that field is a follow-up.
* **D-RLS1-4: accepted.** A kill stops the chain after the arm's evidence is banked; the Master Mind decides before another arm runs.
* **TL-1: OPEN, put to the PI** (the teacher-lane labels are paper-pure DriveRL teacher-simulator signals, so a lean re-implementation would be a different label).

## 1. Before the pod (dev box)

1. **Land the package, then write the manifest** (the launch gate has `require_commit`): `python refe/launch_gate.py ship-manifest --pkg <package root> --commit <landed 40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json`. Files the manifest covers: `refe/ code/ splits/`. Not-yet-landed files REFUSE it: `refe/scorer_finetune.py`, `rl_reward.py`, `rl_fan_dump.py`, `rl_terms_label.py`, `rls1_pod.sh`, `rls1_checks.py`, the gate files and their tests must be on the tip.
2. **Ship to a directory of its own, `/workspace/refe-plan-rls1`** (NOT `/workspace/refe-plan`: P2's `token_ok` re-measures that tree before every `train.py` launch, so adding files there while P2 trains breaks P2). Ship `code/ refe/ splits/ SHIP_MANIFEST.json` plus three things outside the manifest: `eval/` (the chain runs `rl_val_eval.py`, `proxy_eval.py`, `rl_terms_join_identity.py`) and `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012.manifest.json` with its directory of two TSVs (`.gitattributes` there: `-text`; never convert line endings: they are hashed byte for byte). Skip `__pycache__`. The repo is public: no host or port in any file.
3. **Push u400 to the private repo** (the base exists on the dev box ONLY: `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt`, 1,288,489,838 B, md5 `8e382eebb35b3e155862453617e5786e`, sha256 `8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d`, MEASURED 2026-10-12; the dev-box uplink is 1.2 MB/s, so about 18 min): `python refe/hf_egress.py push --repo Sayood/tanitad-refe-rls1 --prefix base --tag u400 --files <that file>` with the token in the environment. The repo is created private by this push. (`model_final.pt` needs nothing: P2's restore put it on the pod, md5-pinned; its sha256 `4246a320...` is MODEL_REGISTRY's.)
4. Provide the teacher-lane directories (TL-1) and `HF_TOKEN` (write-scoped, a pod environment variable; never in a file or in chat).

## 2. Start (one command, detached) and dry runs

```
mkdir -p /workspace/data/rls1 && nohup setsid bash /workspace/refe-plan-rls1/refe/rls1_pod.sh \
  > /workspace/data/rls1/stdout.log 2>&1 < /dev/null &
```
`RLS1_STOP_AFTER=<precheck|env|ckpts|partition|tlane|plan|dump|labels|join|bank|gate_C0|arm_C0|...>` ends the chain after that stage. Cheap dry runs, in this order: `precheck` (seconds), `partition` (checkpoints + partition, minutes), `tlane`, `plan` (image coverage of the 51,000 scenes ON THIS POD, no GPU), then `gate_C0` after the data stages (one preflight and one token).
Refusals are `REFUSED: <reason>` in `/workspace/data/rls1/chain.log` (and on HF: the exit trap pushes it).

## 3. What the chain does, in order (each stage leaves `stage/<name>.ok`; a restart redoes nothing finished)

| stage | what it checks or does | refusal that fires |
|---|---|---|
| precheck | `p2_over`: P2's `chain.lock` not held (flock), **no process in `nvidia-smi --query-compute-apps`** (an unanswerable `nvidia-smi` counts as busy), `ZZP2R_DONE` in P2's `chain.log` (override `RLS1_P2_ABANDONED=1`, the PI's word only); `HF_TOKEN`; every shipped file; a `dd` write test (`RLS1_DD_MB`, default 16 GB; `df` is never trusted) | lock / GPU / P2 unfinished / token / `... missing: ship ...` |
| env | reuses P2's `teacher_env.sh`; backbones present; cgroup cores (v2 `cpu.max`, else v1 `cfs`; MEASURED 11 on the P2 pod); label workers = cores - 1; a 30-min ticker pushes `chain.log` | missing env |
| ckpts | model_final: size + md5 + sha256 against the pin; u400: same, fetched from the repo's `base/` when absent | `... is not the pinned file` |
| partition | manifest sha256 `6a2b5aec...`, `partition_id` `272cfd357b81...`, both TSV sha256 against the manifest, 48,000 / 3,000, 75 val logs (>= 60), log-disjoint; writes `rl_val_logs.txt` | `the partition is not the frozen one` |
| tlane | TL-1 above | `no usable teacher-lane labels` |
| plan | `rl_fan_dump.py --plan-only` on THIS pod; a gap (exit 3) runs the one admissible remedy (`rl_pool_available.py` + the same `rl_partition.py`, seed 20261012), logs OLD and NEW `partition_id`, pushes the remedy to HF and **STOPS**: log both ids in `PREREG_RL_S1.md` section 8, re-pin `rls1_checks.PARTITION`, re-ship (needs `raw/rl_pool/navtrain_token_table.tsv.gz` and `raw/p2_r2_instrument/baseline/navtest_full_tokens.json` shipped) | `image-coverage gap on this pod` |
| dbs, setup | `pod_dataprep.sh STAGES=dbs` (skipped by the marker if present); NAVSIM `_setup()` and `rl_terms_label` import BEFORE hours of labelling | `the labellers cannot start` |
| dump + labels | RL-val first (one shard), then `RLS1_SHARDS` (4) RL-train shards by log (`crc32 % n`) from **model_final** (`--expect-ckpt-sha256`), `OMP_NUM_THREADS=1`; each finished shard is labelled at `nice 19` / `oom_score_adj 1000` while the next dumps: PDM on EVERY proposal (never `--served-only`), audited by `pdm_label_complete.py --all-versions`, then `rl_terms_label.py` (`ZZRELABEL_RLT_DONE n 0 0`) | `dump ... failed`, `PDM labels ... incomplete`, `labels of ... did not finish` |
| **join** | the post-label gate (`rls1_checks.py join-gate`, both sides): the dump IS the partition side (tokens, logs, rank 0, `ckpt_step` 10075, no foreign or duplicate sets); **PDM labels AND `rl_terms` on 100 % of the sets, 64 wide, `label_fires` boolean**; G-REWARD-ID (`gmax > 0 <=> DAC = 0`, `D` exactly 0 where DAC = 1) with **zero** disagreements. Why 100 %: a C run trains on every PDM-labelled set, an `--epo` run only on sets that have `rl_terms`; below 100 % they are not set-identical and C vs R is void | `join gate (train): labels_below_100_percent ...` |
| bank | tar.gz of the six dirs + sha256 MANIFEST + both join reports + the partition manifest -> `Sayood/tanitad-refe-rls1` `data/` (success = the remote listing, `ZZEGRESS_OK rlfan_data`) | `were not banked` |
| per arm | see section 4 | |

## 4. Per arm (order `C0 R0 C1 R1 R2`; arm names, seeds and argv are the gate's: `launch_gate.py` `RLS1_ARM_SEEDS`)

1. `scorer_finetune.py $ARGS --preflight` (G1-G11 incl. the EPO gate G11); `ZZSFT_PREFLIGHT_PASS`.
2. `launch_gate.py run --profile refe_rls1 --arm-name <ARM> --gate-preflight-json <arm>/preflight.jsonl` mints `lg/<ARM>/PASS_*.json`; `token_ok` = `launch_gate.py verify` re-measures argv, tree and data (never a grep). It runs after the minting, before the stage is marked, and again immediately before the trainer starts. **No token, no trainer.**
3. Train (own process group; `--eval-every 250`, `--dump-scores <arm>/scores`): `hf_egress.py watch` beside it (pushes every settled `scorer_states`, halts the run after two failed pushes plus the next checkpoint) to the per-arm repo `Sayood/tanitad-refe-rls1-<arm lowercase>`, and the kill monitor `rls1_checks.py monitor`. After every `score_dump` event it runs the RL-val lookup of R against its control: **K3** (g3/g5 bound breached at two consecutive checks), **K4** (R below the control by > 0.5 PDMS at two consecutive checks), **K5** (R2 only: lane-change rate at the pick above 2 x the label base rate `mean(label_fires)` over RL-val, one check; precision and recall are recorded), **K2** non-finite loss, **K6** more than 6 h. K1 (lam 0 reproduces C) is the preflight's G11 and `test_rl_epo.py` in the gate's G-SUITE.
4. `rls1_checks.py replay` re-judges EVERY tag after the run (a monitor that errored cannot lose a kill); then `rl_val_eval.py` (`--rule navsim_v1_lane`, `--require-full`; R2 adds `--use-k --use-l`): C arm = A0 vs C; R arm = A0 vs its own arm A vs R, and R vs the matched C run when it exists.
5. Egress to the per-arm repo: the checkpoint (A for a C arm; B and A for an R arm), `sft.jsonl`, logs, preflight, token, every `scores_*.npz`, kill and read JSONs. Success is the remote listing.

## 5. Timeline (ESTIMATED; first run on the P2 pod, 11.05 vCPU, one A40)

| stage | estimate | class and source |
|---|---|---|
| precheck, env, ckpts, partition, plan | 0.3-0.6 h | ESTIMATED (u400 fetch 1.29 GB; hashing 2.6 GB) |
| dump 51,000 scenes | **3.3 h** (+ about 0.1 h of model loads) | 0.2313 s/set is MEASURED for the OLD dumper (`dump_fitness.json`); `rl_fan_dump.py` on a pod is UNVERIFIED |
| PDM labels | 4.0-4.4 h at 11 workers | MEASURED rate 212 labels/min at 11 workers (P2R runbook section 6, pixels extracting beside it), versus 1.78 CPU-s/set banked (= 25 CPU-h, 2.3 h): the range is the honest answer |
| `rl_terms` | 1.9 h at 10 workers | ESTIMATED, 1.34 CPU-s/set = 19 CPU-h (`RESULT_RL_TERMS.md`) |
| labels, 63 CPU-h in all, pipelined with the dump | **about 6.5-7 h wall** for dump + labels together (labels are the long pole; 5.9-6.4 h alone) | ESTIMATED |
| join, bank | 0.2-0.4 h | ESTIMATED |
| per arm: preflight 0.1 h + gate 0.15-0.25 h (suite incl. `test_rl_epo.py` ~5 min CPU) + train **4.0 h** (1,000 updates x 14.4 s) + five RL-val evals 0.2 h + egress 0.1 h | **about 4.6-4.8 h** | 14.4 s/update is INHERITED (review 6.2 via PREREG_RL_S1); the gate and eval times are UNVERIFIED. K6 stops an arm at 6 h |
| five arms | about 23-24 h (3 arms, D-RLS1-1: about 14 h) | ESTIMATED |
| **total** | **about 31 h** (about 21 h with the paired design) | ESTIMATED; no pixel extraction competes (P2 finished), so the PDM rate may beat 212/min |

## 6. Watch without SSH, and the markers

HF (private): `Sayood/tanitad-refe-rls1` (`logs/chain.log` every 30 min and at exit, `data/` once, `base/`, `remedy/`), `Sayood/tanitad-refe-rls1-<c0|r0|c1|r1|r2>` (checkpoints and logs while training, `final/` at the end).
`chain.log` markers, in order: `precheck ok`, `env ok`, `checkpoints verified`, `partition verified`, `teacher-lane labels`, `image coverage complete`, `dump ok`, `labels ok`, `join gate PASS`, `dump + labels banked on HF`, per arm `preflight PASS, launch gate PASS (token verified)`, `trainer rc`, `ZZRLS1_ARM_DONE <ARM>`, finally `ZZRLS1_DONE`.
`REFUSED: <reason>` = stopped on purpose. `ZZRLS1_KILL` / `ZZRLS1_ARM_KILLED` = a pre-registered kill (D-RLS1-4). `ZZEGRESS_FAIL` = a push failed; **the evidence is on the pod only: do not delete the pod.**

## 7. Restart, exactly (no addresses)

- **Any interruption, same pod:** run the section-2 command again. Finished stages are skipped; labels and dumps resume by key; an arm interrupted in training restarts at update 0 (`scorer_finetune.py` has no resume; at most about 4.6 h lost) with its old `sft.jsonl`/`sft.log` archived, never appended to.
- **After a `REFUSED`:** read the line, fix the named cause, delete the named marker if the line says so, run the command again.
- **A killed arm:** the evidence is already on HF. Delete `stage/arm_<ARM>_killed` to run the next arm anyway, or remove the arm from `RLS1_ARMS`.
- **A NEW pod after the old one was lost:** ship section 1, set `HF_TOKEN`, run the command. NOT BUILT: an automatic restore of the banked dump and labels. Until it is, fetch `data/rlfan_*.tar.gz` and `MANIFEST_rlfan_sha256.txt` from `Sayood/tanitad-refe-rls1`, check each sha256, `tar -xzf --no-same-owner -C /workspace/data/rls1`, then `touch` the `dump_*`, `label_*`, `plan` and `bank` markers named in `stage/` by a dry run. (The tars hold `sets_*`, `pdm_*`, `terms_*`; NFS refuses `:` in names and `chown`.) A pod lost before the `bank` stage costs the dump and labels (about 7 h); after it, minutes.

## 8. The primary read (not built here)

The arms are judged on **full navtest** (12,146 tokens, 136 logs, paired log-cluster bootstrap), through the stage-3 path of `eval/P2_READOUT_RUNBOOK.md` (`stage2_sft.py` then `stage3_full.py --name <arm> --ckpt <model_sft_*.pt> --expect-ckpt-sha256 ...`), one run per arm checkpoint and one for A0 (u400, expected 88.25; the drift check). The RL-val lookup in `rlval_*.json` is hyper-parameter and early-stop evidence only, never an endpoint. ADOPT / REFUTED / NOT PROVEN are PREREG_RL_S1 section 4; STRATEGIC is PAUSED (PI item 25), reported as paused.

## 9. Evidence status of this build

| claim | status |
|---|---|
| the five argv equal the gate section token for token and the REAL launch gate calls each CLEAN; a C arm with `--epo`, an R arm without it, R0 with K/L, R2 without, a wrong seed, 750 updates, a missing lane trio are REFUSED | MEASURED (`test_rls1_chain.py`) |
| the real `rls1_pod.sh` refuses: P2 lock held, GPU busy, P2 unfinished, no `HF_TOKEN`, missing or wrong model_final | MEASURED (sandbox run of the script itself, fake `flock`/`nvidia-smi`) |
| `token_ok` against tokens minted by the gate's own `mint()` for `refe_rls1`: match, other arm, changed lever, extra flag, other key, edited token, FAIL token, no token | MEASURED |
| join gate below 100 % (PDM, rl_terms, other checkpoint's labels), broken G-REWARD-ID, foreign / missing / duplicate sets, wrong `ckpt_step`, non-boolean label | MEASURED |
| kill rules, monitor and replay with a fake trainer; the default checker on a tiny synthetic RL-val through the real `rl_val_eval.run` | MEASURED on synthetic data; **UNVERIFIED on real scorer dumps** |
| anything on a pod: `flock` behaviour, `ps -o pgid`, the watcher's halt, `os.killpg`, the 300 GB image tree under the data manifest, mint time, `rl_fan_dump.py` numerics and speed on the A40 | **UNVERIFIED** (dev-box dry run only) |
| model_final and u400 pins | MEASURED 2026-10-12 on the dev-box files; re-checked by the test while `E:` is mounted |
