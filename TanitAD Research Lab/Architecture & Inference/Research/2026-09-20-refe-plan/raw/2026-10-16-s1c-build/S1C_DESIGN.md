# S1c design: S1's drivable-area head, trained for a longer budget (launch-ready package, zero A40 spent)

Date 2026-10-16. Author: S1c build subagent (dev box). Tier of every number below: T0-like (head perception on teacher-forced frames, never driving performance).
Evidence classes: MEASURED (ours, artifact named) / INHERITED (another doc, not re-verified here) / ESTIMATED / HYPOTHESIS.
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-s1c-build/` (called `PK` below). Base for every diff: origin/agent/arch-inf-20260803 at `df9871e3` (origin is now `afbc8350`: one commit, no change under `refe/` or `eval/`).

## 0. Headline (integration is NOT done: it needs the Master Mind)

* **The package is landable, but nothing is staged, committed, shipped or launched.** Eleven files change (`PK/PATCH_*.diff`, patched copies under `PK/patched/`, every diff proven to `git apply` onto the tip and reproduce the patched bytes: `PK/PATCHES.json`).
* **Recommended arm: `H1u10k-s0`** = 10,000 updates x 8 rows (accum 1), seed 0, profile `refe_aux_s1c`. **Cost ~4.3 A40-h** end to end (range 4.1 to 5.4), inside the ~7 A40-h scope. Alternative: `H1u2k-s0` (2,000 x 64 rows), ~6.2 A40-h.
* **Expect K2 to FAIL (HYPOTHESIS).** Best in-domain evidence is 0.675 against a 0.80 bar (section 8); the experiment's value is the budget slope on the governing population and the choice of the next lever (`PREREG_S1C_DRAFT.md`).
* **Launch window:** start the chain only after R0c's chain logs `ZZRLS1_QUEUE_CLEAR` (section 6).
* **Landing order (the launch gate has `require_commit`: a token binds a LANDED commit):** (1) apply `PATCH_1..11` onto the tip checkout (or copy `PK/patched/` over it; the bytes are identical) and re-run the suites; (2) **commit and push the eleven files** (the Master Mind's step; I staged nothing); (3) `python refe/launch_gate.py ship-manifest --pkg <plan root> --commit <landed 40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <plan root>/SHIP_MANIFEST.json` (`eval/S1_POD_RUNBOOK.md` section 1.1; the manifest must list `refe/test_s1c.py`: the chain's precheck requires it); (4) ship the WHOLE tree (`code/ refe/ splits/ SHIP_MANIFEST.json` + the two `raw/` paths of the runbook) to `/workspace/refe-plan-s1aux`, file for file. Do not apply hunks onto the pod's older tree (section 6, pod tree).
* **Launch command (after `ZZRLS1_QUEUE_CLEAR`):** `mkdir -p /workspace/data/s1aux && S1AUX_ARMS="H1u10k-s0" nohup setsid bash /workspace/refe-plan-s1aux/refe/s1aux_pod.sh > /workspace/data/s1aux/stdout.log 2>&1 < /dev/null &` (with `HF_TOKEN` in the environment; nothing else is needed: the S1c-only dd size and the lock-only RL-S1 rule follow from the arm list).

## 1. Exact argv, diffed against H1-s0's banked argv

Banked argv: `D:/Projects/TanitAD-artifacts/s1_read_cache/arms/H1-s0/PASS_d257647b7858.json` (`binding.argv`, commit `d257647b`, created 2026-10-15T07:06:20Z). The chain's `arm_args` (extracted verbatim and run under bash by `PK/argv_diff.py` -> `PK/ARGV_DIFF.json`) reproduces that argv **token for token for `H1-s0`** (S1 is unchanged), and for the S1c arms differs only here (MEASURED):

| flag | banked H1-s0 | `H1u10k-s0` (recommended) | `H1u2k-s0` (alternative) |
|---|---|---|---|
| `--max-updates` | 400 | **10000** | **2000** |
| `--batch` | absent (parser default 8) | **8** (named) | **8** (named) |
| `--accum` | absent (parser default 8) | **1** | **8** (named, equals the default) |
| `--eval-every` | 400 | **5000** | **1000** |
| `--dump-scores`, `--out` | `.../runs/H1-s0[/dump]` | `.../runs/H1u10k-s0[/dump]` | `.../runs/H1u2k-s0[/dump]` |

Everything else is identical: `--ckpt` u400, the four label sets, `--lr 1e-4`, `--b-mode expreward`, `--lane-head`, `--require-pdm`, `--seed 0`, `--aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4 --aux-drv-labels .../label_store`, **no `--aux-drv-feed`** (H1 = feed off). Full recommended argv: `PK/ARGV_DIFF.json["H1u10k-s0"]["argv"]`.

**Why the budget flags are NAMED.** S1's argv leaves `--batch`/`--accum` to the parser defaults; a default hides a lever. The profile `refe_aux_s1c` lists `--max-updates --batch --accum --eval-every` as must-name flags and registers the exact values per class (`refe/launch_gate.py` `AUXS1C_BUDGETS`).

## 2. Update count and batch, and the reason

| | rows / update | updates | rows seen | share of one epoch (174,912 train sets) | micro-batches |
|---|---|---|---|---|---|
| S1 (banked) | 64 (8 x accum 8) | 400 | 25,600 | 0.146 | 3,200 |
| **u10k** | 8 | 10,000 | 80,000 | 0.457 | 10,000 |
| u2k | 64 | 2,000 | 128,000 | 0.732 | 16,000 |

* **Recommended u10k, because (a) it is the (updates, batch) pair the dev-box probe measured.** `raw/2026-10-16-scene-tap-a2/A2_RESULT.md` (row U2, MEASURED): S1's head config trained from scratch on ~861 navtest tokens in 5 log-disjoint folds reads held-out IoU [0,40) **0.675** at 10,000 updates x batch 8 against **0.635** at 2,000 (paired **+0.040**, 95 % logs-only CI +0.032..+0.049, probe seed floor 0.0036); in-sample 0.894 vs 0.702. (b) **S1's own head was still learning at update 400:** `aux_bce` fell 0.569 -> 0.432 over updates 50..400 and was still falling ~0.004 per 25 updates over the last 100 (`s1_read_cache/arms/H1-s0/sft.jsonl`, MEASURED). (c) It is **1.9 A40-h cheaper** than u2k. (d) Its micro-batch is S1's micro-batch (8 rows), so peak memory (5.5 GB at S1's preflight) and the loader rate are the measured ones.
* **The honest limit of (a):** the probe is **in-domain** (trained and tested on navtest tokens); S1c trains on navtrain frames and is read on navtest, so the probe is not a prediction of S1c's number. It ranks the budget lever; it does not give the level.
* **u2k is registered as the alternative** (S1's own 64-row update x 2,000): fewer head optimiser steps, more rows, 5x S1's measured window; its larger-batch regime is **untested by any probe** and costs 1.9 h more.
* **Schedule (`refe/scorer_finetune.py:1363-1367`):** `total = min(ceil(len(tr_idx)/batch*epochs/accum), max_updates)`; `lr_at(u)` = linear warm-up over `--warmup` 50 updates then cosine to **exactly 0 at `total`**; the head follows the same shape x `aux_lr/lr` = 2.0 (`:1373`, `:1399-1400`). With `--epochs 1.0` the epoch cap is 21,864 updates at accum 1 and 2,733 at accum 8, so both budgets are below it (the G-DVB schedule row, section 5, proves it per launch).
* **Eval cadence:** u10k evals at step0, update5000, update10000 (3 held-out evals, ~8.0 min each, MEASURED); u2k at step0, 1000, 2000. `scorer_states_last.pt` is rewritten at every eval (`:1418-1423`).

## 3. Is there a head-only mode? No (MEASURED from source)

`refe/scorer_finetune.py` has no head-only flag (`grep -n "head.only\|head_only"` returns nothing). The trunk is frozen (`:619-625`: every model parameter `requires_grad_(False)`, then the scorer modules `True`), but **both scorer arms are always built and stepped**: `optA`/`optB` at `:1361-1362`, stepped at `:1390-1394`, with the head's own `optH` (`:648`, stepped at `:1396-1400`) beside them. `--aux-drv` ADDS the head to a run that still trains the scorer pair. A head-only path would need a new trainer flag, a new G-DVB entry, a new preflight shape and a new argument to the run-check; the saving is ESTIMATED (the trunk forward is shared and dominates, so likely a minority of the 1.24 s per micro-batch). **Not recommended inside this deadline.** Consequence: S1c also produces scorer A/B dumps; they are budget-confounded (section 9).

## 4. Cost model from S1's own logs (`PK/cost_model.py` -> `PK/COST_MODEL.json`)

MEASURED over S1's nine arms (400 updates each): H1 `s_per_update` **9.913 s** (min 9.88, max 9.98, 48 rows; N 9.70, so the head costs +0.21 s per update); one update = 8 micro-batches of 8 rows, so a micro-batch costs **1.239 s**; step-0 eval **8.02 min**, final eval + dumps + saves **8.11 min**; preflight -> trainer start **14.7 / 16.1 / 16.0 min** (H1 arms; +2 min assumed for `test_s1c.py` in G-SUITE). The model reproduces S1's own wall clock: 1.370 h modelled vs **1.362 h** measured (n=9).

An S1c arm at accum 1 runs the SAME micro-batch shape. What S1 cannot identify is the per-optimiser-step overhead `x` (S1 has one accumulation setting); it is bracketed at 0 / 0.05 / 0.5 s (ESTIMATED; Adam over the scorer plus a 2.4 M-parameter head is milliseconds on an A40).

| arm | rows | train h | run h (3 evals) | + preflight/gate gap (17.7 min) |
|---|---|---|---|---|
| 2k x 64 (u2k) | 128,000 | 5.51 | 5.91 | **6.21** |
| 10k x 8, x=0 | 80,000 | 3.44 | 3.85 | 4.14 |
| **10k x 8, x=0.05 (central)** | 80,000 | 3.56 | 3.97 | **4.27** |
| 10k x 8, x=0.5 (worst bracket) | 80,000 | 4.66 | 5.06 | 5.36 |
| S1 itself (check) | 25,600 | 1.10 | 1.37 | 1.67 |

If the gate gap is the 40 min that `BACKLOG.md` measured on P2-s1 (G-SUITE 22 min there), add 0.4 h. **The recommended arm fits the ~7 A40-h scope in every bracket.** The pre-registered check on this model: the arm's logged `s_per_update` (accum 1) must read between 1.24 and ~1.8 s.

## 5. Launch gate, G-DVB and the patches (each lever has its entry)

| lever | where it is registered | where it is checked |
|---|---|---|
| arm classes `H1u10k` / `H1u2k`, seed 0 | `launch_gate.AUXS1C_ARM_CLASS/SEEDS`, profile `refe_aux_s1c` | `auxs1_refusals`: an undeclared arm "must be NAMED (--arm-name)"; `test_s1c.py` section 2 (every budget flag edited, dropped, crossed with the other budget, the seed, the head levers, the arm class: REFUSED) |
| `--max-updates --batch --accum --eval-every` | `AUXS1C_BUDGETS`, `AUXS1C_MUST_NAME` | same, and `required_flags` |
| **the schedule REACHES `--max-updates`** (closes TL-1 for these levers: `max_updates` is only "consumed" in `DVB_KINDS_SCORER`, never observed) | `profile["schedule_reach"]` -> `gate_checks.scorer_schedule_rows` re-derives `min(ceil(train_sets/batch*epochs/accum), max_updates)` from the preflight's data event | deliberate-regression arm `schedule_truncated` (train sets / 4) makes G-DVB FAIL |
| the run ENDS at N, at lr 0, with every eval tag and dump | `s1aux_checks.S1C_BUDGETS`, `run_problems` | `test_s1c.py` section 5 + the end-to-end chain case (section 8) |
| G-SUITE for S1c | `AUXS1C_SUITE` = S1's five scripts + `test_s1c.py` | `launch_gate.py run --profile refe_aux_s1c` (the chain picks the profile with `arm_profile()`, `s1aux_pod.sh:150`) |

No new trainer flag exists, so `DVB_KINDS_SCORER` is unchanged. Patches (against the tip; `PK/PATCH_n_<file>.diff`, copies in `PK/patched/`):

| n | file | change |
|---|---|---|
| 1 | `refe/launch_gate.py` | S1c profile and tables, `auxs1_refusals` generalised (S1 messages unchanged) |
| 2 | `refe/gate_checks.py` | `scorer_schedule_rows` + the `schedule_truncated` regression arm |
| 3 | `refe/s1aux_checks.py` | `S1C_BUDGETS`, `arm_budget`, `eval_tags`, `run_problems` at the arm's budget |
| 4 | `refe/s1aux_pod.sh` | S1c arms, per-arm gate profile, **CR-stripped navtest pin**, **lock-only RL-S1 dependency (S1c-only)**, small dd (S1c-only), launch-window comment |
| 5 | `refe/test_s1c.py` | NEW |
| 6 | `refe/test_s1aux_chain.py` | static checks follow the chain; S1c mutants; `token_ok` per profile |
| 7 | `refe/test_aux_drivable_head.py` | E10 identifies the navtest list by its CR-stripped sha256 |
| 8-11 | `eval/aux_s1_readout.py`, `eval/aux_navtest_forward.py` and their tests | S1c head arms parse; never paired with a 400-update N; the budget contrast S1c-minus-H1 (section 9) |

**The CRLF navtest pin** (backlog item, `raw/2026-10-15-s1-chain-linux-hygiene/`): the chain pinned `0b9c6389...` (the sha of a Windows CRLF checkout); the git blob and any LF ship are the same list without the CRs. Fixed at BOTH sites (the chain, `s1aux_pod.sh:39-40`, and G-SUITE's E10): the pin is the sha256 of the CR-stripped bytes, `3ba1c96488f157fd2ca7c57c83abe439b4c9680ebdb1eadf65c34ac323d42d44`; LF and CRLF forms pass, a different list fails. The second site (E10) was found by running the suite on an LF tree.

### Suites, before and after (exact; dev box, TMP on C:, `PK/logs/`)

| suite (exact command: `PK/tools/run_suites.sh`, venv `tanitad`, `PYTHONUTF8=1`) | BEFORE (tip `df9871e3`) | AFTER (patched) |
|---|---|---|
| `refe/test_launch_gate.py --static --no-banked` | `ZZLG_STATIC_PASS 645` | `ZZLG_STATIC_PASS 645` |
| `refe/test_s1aux_chain.py` | **FAIL on an LF tree** (`ZZS1AUX_REFUSED stage=data`: the CRLF pin); 385 on a CRLF-fixture tree | **`ZZTEST_PASS 404`** on the LF tree (+19: S1c static checks, 7 new mutants, `token_ok` per profile) |
| `refe/test_aux_s1_wiring.py` | `ZZTEST_PASS 20` | `ZZTEST_PASS 20` |
| `refe/test_aux_drivable_head.py` | **FAIL on an LF tree** (E10, the same pin); 68 on a CRLF-fixture tree | **`ZZTEST_PASS 68`** |
| `refe/test_rl_epo.py` / `refe/test_rl_reward.py` | 42 / 31 | 42 / 31 |
| `refe/test_s1c.py` | (new) | **`ZZTEST_PASS 241`** on the dev box (incl. the 17-check end-to-end chain case); **219 + 2 skips** in a pod-like tree built from the ship manifest (the sandbox case and the real navtest list are skipped, not counted, when absent; `logs/podlike_test_s1c.log`) |
| `eval/test_aux_s1_readout.py` (driverl-eval venv) | `ZZTEST_PASS 66` | `ZZTEST_PASS 67` (+1 RED mutation; S1c parse/pairing/budget-contrast asserts) |
| `eval/test_aux_navtest_forward.py` | cannot run outside `D:/Projects/TanitAD` (it asserts the head module lives there) | same; the changed `parse_arm` was exercised directly (S1c names accepted, near-misses refused) |

Logs: `PK/logs/base_*.log` (before) and `PK/logs/final2/*.log` (after, all run on the finished tree). Wall clock on the dev box under 4-way contention: s1aux_chain 11.7 min, aux_drivable_head 10.4 min, test_s1c 4.6 min (2 min alone for its non-sandbox sections), aux_s1_wiring 2.5 min, rl_epo 3.0 min, launch_gate static 2.0 min. The five suites G-SUITE runs for S1 are unchanged in count; S1c adds `test_s1c.py` (~2 min alone, sandbox case skipped on the pod).

## 6. Ordering, launch window, lock behaviour (file:line at the tip `df9871e3`; patched line in brackets)

* **A40 order P2-s1 -> R0c -> S1c -> P6.** P2-s1 (`p2rep_pod.sh`) waits on the S1 lock (`:78`), so it must have started or finished before S1c holds `$DATA/s1aux/chain.lock`; this chain holds it **from its first line** (`s1aux_pod.sh:159-160` [181-182]; `chain.stop` is cleared by `rm -f "$STOPF"` at `:162` [184]; the lock is a plain file held by `flock`).
* **R0c yields to S1c's lock.** `rls1_pod.sh:100-107` `ahead()` returns true while `lock_held "$S1"`; `queue_wait` (`:109`) then logs `ZZRLS1_QUEUE_YIELD`/`QUEUED` and does not take its own lock. R0c logs `ZZRLS1_QUEUE_CLEAR` at `:246-247`. **A S1c chain started earlier would run BEFORE R0c** (R0c holds only `chain.queue` while queued, so S1c's RL-S1 lock check would find the lock free). Start S1c only after `ZZRLS1_QUEUE_CLEAR`. (I did not add a code guard: the brief fixed the dependency as lock-only. A refuse-while-R0c-is-queued guard is ~10 lines plus a test; say if you want it.)
* **S1c's wait for R0c.** `upstream()` (`:68-74` [78-91]) returns 2 while `$P2/chain.lock` or `$RLS1/chain.lock` is held; the precheck lets the CPU stages run beside a training upstream (A4) and `wait_clear` (`:76-90` [92-]) blocks the first GPU use: **poll 30 s (`S1AUX_POLL_S`), 3 consecutive idle polls (`S1AUX_IDLE_POLLS`)** = at least 90 s of GPU idle, once at `gpu_wait` (`:358-359` [380-381]) and again right before the trainer (the in-arm `wait_clear`). `S1AUX_WAIT_MAX_S=0` = unbounded.
* **Lock-only RL-S1 dependency (PI, 2026-10-16), S1c-only** (`S1C_ONLY` is 1 when every arm in `S1AUX_ARMS` is an S1c arm, `:49` [new]): lock held -> waits; lock free and no `ZZRLS1_DONE` -> proceeds and logs `ZZS1AUX_UPSTREAM_NO_DONE rls1 (WARN: ...)` (`:85`); lock free and `ZZRLS1_DONE` -> proceeds silently. S1's own nine-arm chain keeps its refusal (S1-D4). Tests (`test_s1c.py` section 7 and the end-to-end case): lock held waits / free+no DONE proceeds+WARN / free+DONE proceeds / no log proceeds+WARN / P2 lock waits / idle polls = 3 / bounded GPU wait; **deliberate regressions:** restoring the old refusal turns the two S1c cases RED (and nothing else), dropping the WARN is RED, dropping the lock wait is RED. This is the clause that idled the A40 for 8.3 h on 2026-10-14.
* **Only the S1c arm runs.** `S1AUX_ARMS="H1u10k-s0"` (a subsequence of the registered `ARM_ORDER`, `:46`); the default list stays S1's nine, so an S1c launch is never an accident of a default (static check + mutant). Proof: `test_s1c.py` section 8 runs the real chain in the sandbox: `ZZS1AUX_ARM_TRAIN_START` appears once, for `H1u10k-s0`; the only stage marker `arm_*.ok` written is `arm_H1u10k-s0`; one preflight and one trainer start; the evidence lands under `arms/H1u10k-s0/` and no S1 arm directory is touched. S1's nine arms' own markers (`stage/arm_<ARM>.ok`) would skip them anyway.
* **P6 after S1c.** `p6_pod.sh:102-103` REFUSES at precheck while `$DATA/s1aux/chain.lock` is held, and its `wait_clear` (`:426-431`) waits on it: start P6 after `ZZS1AUX_DONE`. No P2REP lock check was added to S1c's `upstream()`: P2-s1 waits for the S1 lock (`p2rep_pod.sh:78`), so S1c waiting for P2REP's lock would deadlock; S1c's guard against a running P2-s1 is the idle-GPU polls.
* **Pod tree: confirmed for S1, UNVERIFIABLE for "now".** S1's arms H2-s0 onward were minted and run under commit `d257647b` (tree sha `fb65475c...`, `arms_manifest.jsonl`: 2026-10-15T05:21Z to 14:01Z; N-s0..s2 under `6f9756db`). The tip's `s1aux_pod.sh` is byte-identical to the `d257647b` version (it is absent from `git diff d257647b FETCH_HEAD`), but `launch_gate.py`, `gate_checks.py` and `scorer_finetune.py` differ at the tip by R0c additions. Whether the pod's `refe-plan-s1aux` still equals `d257647b` today cannot be read from here. **Ship the whole patched tree**; the gate re-measures it and `SHIP_MANIFEST.json` must be regenerated.

## 7. Disk and the dd write test (NFS, lazy reclaim)

* **Where:** `s1aux_pod.sh:361-365` [385-389]: `DDMB="${S1AUX_DD_MB:-24576}"` then `dd if=/dev/zero of="$DATA/.dd_test_s1" bs=1M count="$DDMB" conv=fsync`. It runs **AFTER `wait_clear`** (`:358-359`), i.e. after R0c is over, and the file is removed straight after the size check. The default for S1's nine arms is 24 GiB (nine arms at ~2.7 GB).
* **Recommendation: `S1AUX_DD_MB=5120` (5 GiB), now the built-in default when every arm is an S1c arm** (`DD_DEFAULT_MB`, `:385` [new]); override with the env var `S1AUX_DD_MB`. Need: `model_sft_A.pt` + `model_sft_B.pt` 2 x 1.29 GB (the chain's `MODEL_B_MIN_BYTES` and u400's 1,288,489,838 B), `aux_drv.pt` 29.3 MB (MEASURED), 3 evals x ~69 MB of dumps (MEASURED per tag from the cache: head_pred 32.8 MB + 3 score files ~12 MB), `scorer_states_last.pt` (UNMEASURED, bounded <= 0.5 GB), plus one transient 1.3 GB (`atomic_save` writes beside the target): **~3.3 GB steady, ~4.6 GB peak, so 5 GiB covers it** (ESTIMATED from measured file sizes). At the pod's ~35.8 GB free (coordinator's figure, INHERITED) the 24 GiB default would claim two thirds of it and be reclaimed lazily; 5 GiB claims 14 %. I changed neither P6's nor R0c's files.

## 8. Reachability (CPU, dev box; the 8 GB GPU was not used)

Real `scorer_finetune.main()` with the real `AuxDrivable` head (2,440,450 parameters) on a 256-wide stub trunk and a synthetic label store, at the SHAPE of each budget. Evidence: `PK/REACHABILITY.json` (written by `S1C_EVIDENCE_OUT=... python refe/test_s1c.py`).

* **Head gradient non-zero:** G12 per-group |grad| at the declared weight: q 1.90, pos_mlp 7.52, blocks.0 480.9, blocks.1 484.6, norm 1.01, out 16.0; **weight-0 control exactly 0.0 on all six groups.**
* **Update counter reaches N:** start event `updates` 50 / `micro_per_update` {1, 2} / `batch` 2; done 50; the head optimiser's Adam `step` = **50 on all 43 head tensors** (43 of 43 with state); every head tensor moved; eval tags step0 / update25 / update50.
* **Schedule ends at N:** last train row update 50 of 50 at **lr 0.0** (head `aux_lr` 2.1e-7 = the last step's rate).
* **Deliberate regressions go RED** (each snippet occurs exactly once): `--max-updates` not honoured (start/done/last row/Adam steps all wrong), `--epochs 0.2` (the TL-1 silent early stop: the run ends at 10 of 25, rc 0), the schedule not ending at N, `optH.step()` removed (Adam steps [] on 0 of 43, 43 tensors never moved), the aux loss dropped from backward (G12 rc 3, no PASS file), the head cut from the loss (rc 3, no PASS file). The unmutated control reads clean.
* **End to end:** the real chain in the sandbox runs ONLY `H1u10k-s0`, mints and verifies under `refe_aux_s1c`, banks `arms/H1u10k-s0/{aux_drv.pt, dump/head_pred_update5000.npz, ...update10000.npz, ...}`, proceeds with the WARN when RL-S1 never logged DONE, and a trainer that stops at half the updates (rc 0, `ZZSFT_DONE`) is REFUSED by the run-check and banks nothing.
* **Not proven here:** a real-GPU run (that is the A40's job), the real trunk, and the 10,000-update wall clock (modelled, section 4).

## 9. Eval plan (post-run, dev box; STRATEGIC PAUSED)

1. **Fetch** `arms/H1u10k-s0/` from `Sayood/tanitad-refe-s1aux`: `aux_drv.pt`, `sft.jsonl`, `run_check.json`, `dump/head_pred_{step0,update5000,update10000}.npz`, `dump/scores_*`.
2. **Identity control C1 (free):** the arm's `step0` `aux_eval` IoU must equal S1's H1-s0 step0 (**0.1586791569036157**, same seed, same init, same frozen trunk and held-out frames) to 1e-3, else the arm is not paired with S1 (environment drift).
3. **K2 on the C3 set (governs, as in S1 prereg 4.1):** `aux_navtest_forward.py --heads H1u10k-s0=<aux_drv.pt> H1-s0=<S1 aux_drv.pt>` (one frozen-trunk forward, cached ctx), then `aux_s1_readout.py perception --pred H1u10k-s0=... H1-s0=... --pred-init H1u10k-s0=<seed-0 init head_pred> --labels-set navtest --dac-csv <u400 csv> --c3-table ... --c3-tokens ...`. Reads P1 [0,40) all frames and the u400-DAC-zero split, P2, P3, the init floor, and **`head_fit_pairs["H1u10k-s0-minus-H1-s0"]`** (the budget contrast, paired on the same log resamples; patch 8 adds it, with a test).
4. **K2 on the natural navtest population:** re-weight the same predictions with the SHIFT_LOCATE weights (`raw/2026-10-16-aux-shift-locate/b1_decompose.py` section S5: failure stratum 4.36 %, navtest's own city mix, ESS reported). **Open item: a ~10-line driver for the S1c arm is not written** (`shf_common.navtest_pred_path` is keyed to S1 arms); it is a CPU step at read time.
5. **Held-out (24 navtrain logs) P1** from the arm's own `head_pred_update10000.npz`: informational (navtest governs).
6. **Estimator:** paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927 (`proxy_eval.boot`'s draw scheme). It answers "would another draw of LOGS say this?" only. Training-run variance needs the replicate (seed 1, ~4.3 A40-h, a PI decision). Inference variance: the head is deterministic (ASSERTED, not measured).
7. **The four families, per family (never pooled):**

| family | status for S1c | reason |
|---|---|---|
| LONGITUDINAL (EP, TTC of the pick) | NOT COMPUTED paired; arm B's pick is read DESCRIPTIVELY against the three 400-update N arms, stamped budget-confounded | no budget-matched N arm: arm B trains 25x the updates at batch 8, so a difference is not attributable to the head. An `N-u10k` control (~4.2 A40-h) would make it so. |
| LATERAL (DAC of the pick) | same | same |
| TACTICAL (pick / random / best-of-64) | same | same |
| STRATEGIC | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted, not extended | - |

The navtest-planner families (speed/along-track/heading/cross-track/curvature/yaw-rate/headway/decision accuracy) need the planner seam and are not computed at S1 either (S1 prereg 4.2).

## 10. Open questions and risks

1. **K2 is unlikely to clear (HYPOTHESIS):** S1 H1 reads 0.496 / 0.521 / 0.486 on C3 and 0.561 / 0.577 / 0.554 on the natural navtest population; the probe's best in-domain read is 0.675. Say so in the PI summary before the A40 is spent; the case for spending is the budget slope plus the decision it unlocks.
2. **Launch-window guard** is procedural, not mechanical (section 6).
3. **SHIP_MANIFEST.json** regeneration and the ship location are the Master Mind's.
4. **Gate gap:** S1's preflight->start was 14.7-16.1 min (MEASURED); `BACKLOG.md` measured 40 min on P2-s1 (G-SUITE 22 min). `test_s1c.py` adds ~2-4 min to G-SUITE (its sandbox case is skipped on the pod). Worst case +0.4 A40-h.
5. **Batch 8 for the scorer arms:** `--lr 1e-4` was set for 64-row updates; at 10,000 steps of 8 rows the scorer arms A/B see 25x more, noisier steps. The head is unaffected (H1: feed off), but arm B's picks are not comparable to S1's.
6. **`S1_POD_RUNBOOK.md`** still says `test_s1aux_chain.py` `ZZTEST_PASS 348`; the measured count was 385 before and is now higher (section 5). Not edited.
7. **Natural-population K2 driver** and the **`N-u10k` control** are named, not built.
8. **Pod state (INHERITED, not verifiable here):** S1c's CPU stages (`fetch`, `data`, `relabel`, `bank_labels`) are cheap only if the pod still holds `/workspace/data/s1aux/stage/*.ok`, the u400 checkpoint, the held-out bank and the label store from S1 (S1 finished there on 2026-10-15). On a recycled pod `relabel` costs hours (S1-D2) beside R0c's training. Check `ls /workspace/data/s1aux/stage/` before the launch.
9. `aux_navtest_forward.py`'s own test needs the `D:/Projects/TanitAD` checkout (it asserts the head module lives there), so it cannot run in a temp tree: its change (one regex) was exercised directly.

## 11. Deliverable manifest

See the table at the end of the hand-back and `PK/MANIFEST.md`. Everything is in ONE location (this dev box, `C:/Users/Admin/AppData/Local/Temp/s1c/` and the work tree `C:/Users/Admin/AppData/Local/Temp/s1c_work/`); **nothing is in the repo, on a pod, or in a worktree**.
