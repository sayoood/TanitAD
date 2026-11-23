# DRY_MINT_RK: CPU dry mint of the RK kit on the LoRA pod (card PREREG_V8F_KINLAB section 4: the full-bank census, both seeds)

Evidence class: MEASURED on the pod CPU, 2026-11-23 01:22-02:21Z (`dry_results/dry.log`; scratch root `/root/rollhist/dry_rk/root`, deleted afterwards, see `DELETIONS.log`). Driver: `dry_rk_mint.sh` (run from `/root/rollhist/kit_rk/`).
Guards held: nice 19, OMP/MKL 2 threads, CUDA hidden; container `total_rss` 9.6 GB before (limit 40 GB), peak sampled 16.3 GB; the selfcheck process peaked at 5.2 GB RSS (limit 12 GB); 28.1 and 28.3 min wall (limit 30). No GPU, lock (`/root/gpu_queue.lock`), marker (`/root/N1DW_READY`, held by the ADD123 chain throughout) or foreign process was touched; nothing was written outside `dry_rk/` and `kit_rk/`.

## What ran, on the real inputs
1. `rk_tree_prep.sh` (scratch root): tree_rk = the real tree_ra (df250a00, `checked 3415 of 3415 bad 0`) + `rk_delta.tar`, then `checked 3418 of 3418 bad 0` against `all_blobs_rk.txt`: **the real tree delta, blob by blob**.
2. `rk_argv.py render`: RL_s0 / RL_s1's real argv (only the two `/root/rollhist` paths, the bank and `--out`, rewritten to the scratch root) -> RK_s0 / RK_s1 argv; `ZZ RK-RENDER PASS`; `rk_argv.py verify` passes (exactly the label value and `--out` differ; bank sha256 = sha256(bank.json) of the real banks `e5e44c01…` / `bde6566e…`).
3. The stock pair check on the argv minus the two label tokens against B0: `ZZ ROLLHIST-PAIR PASS` (both seeds).
4. The trainer's `--selfcheck-only` of the RK argv, **CPU, full banks** (`--device cpu` is the ONLY difference from the launch argv; the real init checkpoints UE_s0.A / UE_s1.A): rc 0 at both seeds, `declared_vs_built` PASS, `dagger_probe.pass` True.
5. `rk_gate_record.py` mint + verify on the real selfcheck: MINTED and `VERIFY OK` at both seeds (the census share is in the record).
6. `chk_arm_rk.sh mint` in its DRY form (`DRY_GD` containing `/dry_`, `DRY_SC` = the selfcheck above): **21 of 21 checks pass at both seeds** (table below), `TOKEN-WRITTEN`.
**Not real:** the G-SUITE at 0ca6f49d has not run. The scratch root held a PLANTED `gsuite_result_rk.json` (RA's result, commit set to 0ca6f49d, key `PLANTED_FOR_DRY_MINT_ONLY`) and a planted `done_rk/tree.done`; so the `G-SUITE` row below and the gate record's G-SUITE / tree-marker checks stand for the real chain steps, not for a measured G-SUITE. Both planted files were deleted with the scratch root.

## Result: the census bar (card section 4, relabelled share of ok records >= 0.5): MET at both banks
| bank | ok records | relabelled (kin_v1) | share | bar | tick 1 share | tick 2 share |
|---|---|---|---|---|---|---|
| s0 (sha256 e5e44c01…) | 1,490,036 | 953,301 | **0.6398** | 0.5 | 0.6544 | 0.6212 |
| s1 (sha256 bde6566e…) | 1,478,090 | 945,608 | **0.6397** | 0.5 | 0.6543 | 0.6209 |

Both match the one-shard census 3 (0.6387 on shard 0 of s0, `KINLAB_CENSUS_3.md`). Identity-rule records: 0 at both. Refusals (pooled): not_merged 20.8 % / 20.8 %, path_exhausted 7.8 % / 7.7 %, infeasible 4.8 % / 5.1 %, reference refusals (speed_*) 2.7 % / 2.4 %; no non-finite. Per tick and reason below. The gate record's share is the same number (`census` in `RH_GATE_RECORD.json`, re-derived by `verify`).

## Cost finding (reported, not decided here): the kin labeller wall
`selfcheck.json` `info.dagger.speed_label.wall_s`: **1,631.5 s (s0) and 1,643.1 s (s1)** on this pod CPU at nice 19 while it trains (single thread, ~100 % CPU). The same field at RL (speed_v1, MEASURED in `/root/rollhist/runs/RL_s*/selfcheck/selfcheck.json`): 191.8 s and 199.7 s. The trainer builds the labels at load, so the labeller runs **twice per arm under the GPU lock** (the selfcheck, then the training run), in an otherwise CPU-idle phase. Card section 6 priced it as "once at load, on the CPU" inside 0.6-0.75 pod-h.
* RL chain (MEASURED `rh_rl_chain.log`): lock 08:58:55 -> 09:43:31 = 44.6 min = 0.74 pod-h; per arm 22.3 min.
* RK per arm = 22.3 min - 2 x 3.2 min (speed_v1 labeller) + 2 x T_L. With T_L = 27.3 min (this dry, niced, shared pod): 70 min per arm, **2.3 pod-h** for the chain. With T_L = 15 min (ESTIMATED from the dev-box census rate, 0.60 ms per ok record, `KINLAB_CENSUS_3`; an un-niced pod with no training next to it): 46 min per arm, **1.5 pod-h**. Both are above the card's 0.9 pod-h line ("a projection above 0.9 pod-h goes back to the PI").
* Options for the MM (none implemented; each changes code or the gate): label once and reuse in the training run; parallelise `kin_label_bank` over clips (the pod has ~15 cores; a code change, so a new commit and G-SUITE); or take the PI's go at the higher cost.

### RK_s0 (bank /root/rollhist/bank/s0, full bank)

* `declared_vs_built` PASS; `dagger_probe.pass` True; `n_label_kin` 23; `rl_label_rows_changed` 33; `grad_diff_all` 11683.693359375; `control_p0_grad_diff` 0.0
* **relabelled share of ok records (bar 0.5): 0.6398** (953,301 of 1,490,036); `kin_status_counts` = {"kin_emitted": 953301, "refused_not_merged": 310272, "refused_infeasible": 70901, "speed_refused_infeasible": 22624, "refused_path_exhausted": 115777, "speed_path_exhausted": 17161}
* labeller wall (CPU, nice 19, shared pod): `wall_s` 1631.5 s; n_eligible 1490036, n_eligible_after 1490036, n_labelled 1490036, n_emitted 1450251

| tick | ok | relabelled | identity | relabelled share | refused by reason |
|---|---|---|---|---|---|
| 1 | 833,179 | 545,255 | 0 | 0.6544 | {"refused_not_merged": 159859, "refused_path_exhausted": 67586, "refused_infeasible": 44612, "speed_path_exhausted": 6995, "speed_refused_infeasible": 8872} |
| 2 | 656,857 | 408,046 | 0 | 0.6212 | {"refused_not_merged": 150413, "refused_path_exhausted": 48191, "refused_infeasible": 26289, "speed_path_exhausted": 10166, "speed_refused_infeasible": 13752} |
| pooled | 1,490,036 | 953,301 | 0 | 0.6398 | {"refused_not_merged": 310272, "refused_path_exhausted": 115777, "refused_infeasible": 70901, "speed_path_exhausted": 17161, "speed_refused_infeasible": 22624} |

* census `relabelled_share_of_ok` (labeller's own): 0.6397838709937209
* gate record: MINTED (pass True, census 0.6397838709937209)

**checks.json (study-side dry mint), all checks:**

| check | pass | note |
|---|---|---|
| C-ARGV | True |  |
| C-RH-PAIR | True |  |
| C-MERGE | True |  |
| C-MD5 | True |  |
| C-STEPS | True |  |
| C-NCLIPS | True |  |
| C-CHUNK | True |  |
| C-TRAIN-SHA | True |  |
| C-DISJ | True |  |
| C-HOLD | True |  |
| C-MASK | True |  |
| C-JOIN-EQ | True |  |
| G-HYG | True |  |
| G-LIVE | True |  |
| C-BANK | True |  |
| G-DAGGER | True |  |
| C-RH-BANKGEN | True |  |
| G-CLOCK | True |  |
| G-EVAL | True |  |
| G-CKPT | True |  |
| G-SUITE | True |  |

### RK_s1 (bank /root/rollhist/bank/s1, full bank)

* `declared_vs_built` PASS; `dagger_probe.pass` True; `n_label_kin` 33; `rl_label_rows_changed` 45; `grad_diff_all` 50019.17578125; `control_p0_grad_diff` 0.0
* **relabelled share of ok records (bar 0.5): 0.6397** (945,608 of 1,478,090); `kin_status_counts` = {"kin_emitted": 945608, "refused_not_merged": 307371, "refused_infeasible": 75956, "speed_refused_infeasible": 21390, "refused_path_exhausted": 113427, "speed_path_exhausted": 14338}
* labeller wall (CPU, nice 19, shared pod): `wall_s` 1643.1 s; n_eligible 1478090, n_eligible_after 1478090, n_labelled 1478090, n_emitted 1442362

| tick | ok | relabelled | identity | relabelled share | refused by reason |
|---|---|---|---|---|---|
| 1 | 832,885 | 544,990 | 0 | 0.6543 | {"refused_not_merged": 157699, "refused_path_exhausted": 66208, "refused_infeasible": 49049, "speed_path_exhausted": 6862, "speed_refused_infeasible": 8077} |
| 2 | 645,205 | 400,618 | 0 | 0.6209 | {"refused_not_merged": 149672, "refused_path_exhausted": 47219, "refused_infeasible": 26907, "speed_path_exhausted": 7476, "speed_refused_infeasible": 13313} |
| pooled | 1,478,090 | 945,608 | 0 | 0.6397 | {"refused_not_merged": 307371, "refused_path_exhausted": 113427, "refused_infeasible": 75956, "speed_path_exhausted": 14338, "speed_refused_infeasible": 21390} |

* census `relabelled_share_of_ok` (labeller's own): 0.6397499475674688
* gate record: MINTED (pass True, census 0.6397499475674688)

**checks.json (study-side dry mint), all checks:**

| check | pass | note |
|---|---|---|
| C-ARGV | True |  |
| C-RH-PAIR | True |  |
| C-MERGE | True |  |
| C-MD5 | True |  |
| C-STEPS | True |  |
| C-NCLIPS | True |  |
| C-CHUNK | True |  |
| C-TRAIN-SHA | True |  |
| C-DISJ | True |  |
| C-HOLD | True |  |
| C-MASK | True |  |
| C-JOIN-EQ | True |  |
| G-HYG | True |  |
| G-LIVE | True |  |
| C-BANK | True |  |
| G-DAGGER | True |  |
| C-RH-BANKGEN | True |  |
| G-CLOCK | True |  |
| G-EVAL | True |  |
| G-CKPT | True |  |
| G-SUITE | True |  |

