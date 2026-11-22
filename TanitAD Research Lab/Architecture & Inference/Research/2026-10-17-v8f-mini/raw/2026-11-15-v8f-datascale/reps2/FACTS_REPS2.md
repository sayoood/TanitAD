# FACTS_REPS2 (builder agent, 2026-11-22): what UE_s1.A ran on, what the checker admits, how the GPU is shared, the measured walls

Every line carries its class. MEASURED = read by this builder from the pod / local file named. INHERITED = quoted from a record, not re-verified. Pod = the LoRA pod, read-only reads only.

## 1. Code and tree of UE_s1.A, and where the gate token binds them
* MEASURED (`/root/dscale10k/gate/tokens/UE_s1.A/gate_token.json`, minted 2026-11-19T05:33:22Z): commit `cdbff971fa6dad4d20b937fe8abda6cb9f622adb`; argv sha256 `af94ad20935138893160c75ad0fb34b507fd5cf996780fdde13d1dc8f81f2337`; tree manifest sha256 `7ccc204d50da...` (pod tree `/root/dscale10k/tree`, which has no `.git`: the commit is a CLAIM passed on the checker's command line, the tree manifest is the measured part); trainer sha256 `f97f0b43ecfe...` (`tree/stack/scripts/v8f_mini_train.py`, re-read today: same prefix); checker sha256 `ac37c1b61aa5...` = `blockreps1/study/dscale10k_check.py`; 17 required checks (C-ARGV, C-CHUNK, C-DISJ, C-HOLD, C-INIT, C-JOIN-EQ, C-MASK, C-MD5, C-MERGE, C-NCLIPS, C-STEPS, C-TRAIN-SHA, G-CKPT, G-CLOCK, G-EVAL, G-HYG, G-LIVE, G-SUITE).
* Data bound into the token (sha256 prefixes): `--join-train` merged/join_union.npz `7d2be0e6`; `--lead-train` lead_union.npz `e96c2d5e`; `--speed-train` speed_union.jsonl `fd808a26`; `--tracks` native_tracks_union.pt `90229524`; `--v9-train` v9_labels_union.npz `c630dab0`; `--p10` p10_prior.pt `14a062a9`; cache `train/manifest.jsonl` `4d2d4479`; merge_record `ba56af0c`; tenk_v9 `74648b91`; the dry record and the evidence file of the arm.
* Where the checker binds them (`blockreps1/study/dscale10k_check.py`, MEASURED, equal to `reps1/pod_copy/dscale10k_check_reps1.py`, sha256 `ac37c1b6...`):
  * `:855` `binding()` builds `{"arm","stage","commit","argv_sha256": argv_sha(toks),"data": d,"tree_manifest_sha256","trainer_sha256","checker_sha256","merge_tool_sha256","card_sha256_prefix"}`; `:848` `data_binding()` (`:793` DATA_BIND_FLAGS, `:828` the cache manifest) hashes every data file.
  * `:876-899` `token_problems()` refuses unless the token body is unedited (`body_sha256`), every required check passed, and each bound key (commit, argv sha256, data, tree, trainer, checker, merge tool) equals the current value (`:897`).
  * `:1024-1036` `run_arm()`: any problem prints `REFUSED <arm>` and returns **rc 2 before the trainer starts** (`:1034-1035`).
  * The commit is fixed in the wrapper: `reps1/chk_arm_reps1.sh:13` `UE_s1) TREE=$D/tree; COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb`. REPS2 uses the same tree and commit (`chk_arm_reps2.sh`).
* INHERITED (LOOP_STATE_2026-11-02.md line 3503): "BL_s0.A and UE_s0.A gate tokens both bind cdbff971."

## 2. Does the checker admit seeds 2 and 3 today? No.
* MEASURED: `dscale10k_check.py:116-117` (`argv_diff_problems`): `if arm not in ARMS: return [f"unknown arm {arm!r}"]`. `ARMS` (`:60-75`) has `UE_s0`, `UE_s1`, `UM_s0`, `UM_s1`, `B_rep` (seed 2, a B-set arm), `B_s3`, `U_s3` ... but no `UE_s2` / `UE_s3`; the CLI also refuses them (`:1045`/`:1050` `choices=sorted(ARMS)`). Test `test_reps1_refuses_the_new_arms` pins this.
* The `--seed` pin itself (`:139`, `pinned = {"--seed": [str(spec["seed"])]}`) would hold the seed once the arm exists, but `--out` is only "allowed to differ" (`:136`), not pinned, and the B_s0-relative rule lets the union data flags differ (`DATA_FLAGS`), so "UE_s0 with exactly two changes" was NOT enforced by the checker for any arm. The REPS2 patch adds that rule (`rep_ref_problems`), see HANDOFF.

## 3. How the reps1 queue held the GPU lock and the marker
* MEASURED (`reps1/queue_reps1.sh:17`): `exec flock "$LOCK" bash "$CHAIN"` with `LOCK=/root/gpu_queue.lock`: the whole chain ran under the lock; it waited for nothing else. **reps1 used NO marker**; it predates the `/root/N1DW_READY` protocol.
* MEASURED (`/root/rollhist/rh_rl_chain.sh`, `rh_rl_marker_watch.sh`): the current protocol. The chain does `exec 9>/root/gpu_queue.lock; flock 9`, holds fd 9 to the end, logs `GPU queue lock held to the end of the chain`. The watcher (started BEFORE the chain) does `touch /root/N1DW_READY` (the other session's next slice then skips; its running slice finishes), reads only chain-log lines written after it started, and removes the marker when it sees that line, when the chain ends first (`ZZ ROLLHIST-RL ...`), or when no chain process exists (after 120 s); cap 30 h.
* Pod state at build time (MEASURED 2026-11-22 ~19:2xZ): `/root/N1DW_READY` absent; `/root/gpu_queue.lock` file present; GPU 7,158 MiB used (TanitADRuled slice running under `flock /root/gpu_queue.lock`); `chains/` empty (no STOP). Pod free disk **55 GB** (`df /root`), BELOW the chain's 60 GB guard; one stage-A run directory is 84 MB (UE_s1), `tmp/` 8 KB, so the 60 GB guard is conservative (see HANDOFF). Container memory limit 70,999,998,464 B, total_rss 3.94 GB (cgroup v1).

## 4. UE_s1.A measured walls (UE_s0.A for contrast)
| step | UE_s1.A | source |
|---|---|---|
| prep (dry build 4 m 10 s, init, reach probe ~15 s, evidence, mint ~1 m 20 s), UE part only | 05:27:38 -> TOKEN-WRITTEN 05:33:22 = 5 m 44 s | `prep_run.out` (reps1) |
| trainer start + data build + its own selfcheck | `run` 05:45:42 -> `selfcheck.json` 05:49:57 = **255 s** | chain log, file mtime |
| training, 61,568 steps | **11,561.04 s** (3.21 h), 0.1878 s/step | `runs/UE_s1/A/summary.json` `seconds` |
| summary check / post (zero-grad) gate | 3 s / 2 s (post.json: registered 252, with_adam_state 252, problems []) | file mtimes |
| bank to the private HF repo (sha read back) | ~18 s (`ARM_A_DONE` 09:02:43 -> `ARM_DONE` 09:03:01) | file mtimes |
| whole arm, chain stamps | 05:45:42 -> 09:03:01 = 11,839 s = **3.29 h** | `blockreps1.log` |
| UE_s0.A, for contrast (slower pod state) | 13,803.9 s = 3.83 h | `runs/UE_s0/A/summary.json` |
* ESTIMATED for REPS2: 2 arms x 3.3-3.8 h + prep under the lock ~0.35 h (two dry builds and two init records are CPU, 4 min each) = **about 7 to 8 pod-h**, i.e. 3.5-4 h per arm, inside the 3-5 h the fork guessed to the PI.
* Checkpoint ckpt_last.pt: 86,970,901 B (87 MB). `ckpt_every` default 500 (`v8f_mini_train.py:295`).

## 5. Resume (quoted at the code of record, `/root/dscale10k/tree/stack/scripts/v8f_mini_train.py`, sha256 prefix `f97f0b43`)
* `:3499` `if os.path.exists(ck_path) and not a.no_resume:` loads model, optimiser, step, epoch, batch-in-epoch; `:3501` `if ck["argv_sha"] != argv_sha(argv): ... raise SystemExit("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")`; `:3661` `if step % a.ckpt_every == 0: save()` (every 500 steps); `:3537-3538` `torch.save(blob, ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)` (atomic, never a torn ckpt_last.pt); docstring `:4` "writes summary.json {done: true} ONLY at the very end".
* Consequence used by `blockreps2.sh`: a directory with `ckpt_last.pt` and no `summary.json` is a killed run and is resumed by the trainer itself through the same token-checking runner; reps1's chain refused such a directory (`blockreps1.sh` change 6).
* UNVERIFIED (needs the GPU, not run): that a resume reproduces the uninterrupted weights bit for bit on THIS argv. INHERITED: `gate/resume.json` (G-CKPT, `resumed_equals_straight_bitwise`) was measured for the tree at block 1 and is bound in the token as G-CKPT. Same-seed replicate is bit-identical on box 2 for a wm-only trainer, NOT for the full DAgger trainer (memory); this chain has no DAgger.

## 6. The 2026-11-19 delivery route for UE_s0 and UE_s1 (INHERITED, quoted)
* `E:/Projects/TanitAD-artifacts/dagland/LOOP_STATE_2026-11-02.md` line 3373 (2026-11-19 11:13Z): "Sent: UE_s0/A (2fd4b747) and UE_s1/A (f13df8ce) on <account>/tanitad-v8f-dscale10k-runs; commit cdbff971; eval cache sfresh_inputs/cache/eval (manifest 78631f83, 141 clips); native_tracks.pt 1db431f1". I.e. the route was the **private HF dataset repo written by the chain's banker** (`dscale_banker.py`: uploads `<arm>/A/ckpt_last.pt` + provenance in one commit, reads every file back from that commit, requires the hub LFS oid to equal the sha256, then writes `bank_ok/<arm>.A.sha256`); the coordinator fetched it onto the fork pod. Line 3415: banked UE_s1/A `f13df8ce` = pod.
* NOT FOUND: a record of a direct pod-to-pod or relay copy for UE_s0 / UE_s1 (searched LOOP_STATE and COORD_V8F_2026-11-19.md). The fork pod's HF credentials were not verified by this builder. So `deliver_reps2.sh` keeps the hub copy (the chain's banker writes it) AND relays the file by pod -> dev box (`lorapod/pcat.sh`) -> fork pod (`v8pod` ssh), sha256 checked at all three places, destination `/workspace/v8proof/integ10k/ckpt/UE_s{2,3}/A/` (COORD_V8F_2026-11-19.md line 372), with an arm record JSON.
