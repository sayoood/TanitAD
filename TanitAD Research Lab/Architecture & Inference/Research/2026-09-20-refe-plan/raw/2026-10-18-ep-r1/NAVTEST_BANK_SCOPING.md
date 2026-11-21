# EP-R1b (2) -- the navtest pre-head FEATURE BANK: does it exist, and what is the forward that makes it

Date 2026-10-19. Author: Architecture & Inference agent (EP-R1b). Status: SCOPING. Nothing here was run on a GPU, on the S0 bank, or on any outcome array.
Evidence classes: **MEASURED** (ours, artifact path given), **INHERITED** (another package, not re-verified), **ESTIMATED**, **HYPOTHESIS**.

## 1. Answer

**No navtest bank of P2 `score_head` input features exists.** The shared navtest look (MM_DECISIONS_EPR1 D1: SR-TTC alone, EP-R1 alone, the joint, ONE open) cannot be applied until one is built,
because a refit row acts on `feat`: `new logit = stored + feat . (row - P2 row) + (b - P2 b)` (`epr1.apply_row`). Logits alone cannot stand in for it: the head maps 256 features to 6 logits, which is not invertible.

**The forward that makes it:** `srttc_s0_dump.py` in its **fan path** (no `--sets-dir`, **not** `--lean`) with `--feat-out`, **GPU**, checkpoint
`D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` (P2; sha256 `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51`, MEASURED 2026-10-19), once per landed chunk of the full-navtest table (14 chunks, 12,146 tokens, 136 logs).
**Dev-box GPU: 5.43 h** (MEASURED rate of the landed forward, below) behind S0. **Disk: 398 MB features + ~0.3 GB fan dump = ~0.7 GB** (free on D: 120.8 GB, MEASURED 2026-10-19).

## 2. The search (absence at one place is not absence: two names, several paths)

Names tried: `*prehead*`, `*_feat*.npz`, `*feat_out*`, `*features*.npz`, `*feat*.npy`, `*score_head_in*`, and the array key `prehead_features` (header-listed). Paths tried: both mirrors of the data tree
(`D:` and `E:/Projects/TanitAD/data/refe_navtest/` -- different inode ids, identical listing), the full-table package, the FREEPOINT package, `C:/Users/Admin/freept/`, SR-TTC's S0 run tree, and the artifacts root.

| location | what it holds | features for navtest tokens? |
|---|---|---|
| `D:/E:/Projects/TanitAD/data/refe_navtest/` (frames, seams, score, points, proptable, token lists) | frames, W3 seams (`stage2_p2_full.npz` among them), token lists | **no** (`find` -> 0 hits) |
| `D:/Projects/TanitAD-artifacts/p2_full_table/` : `table_p2_full.npz`, `dumps/dump_main_c00..c13.npz` | stored logits, executed proposals, pick, outcomes | **no.** Header of `dump_main_c00.npz`: `token, fingerprint, proposals [990,64,8,3], logits [990,64,6], pick, select, rule, repair_last_heading, ckpt, sampling` -- no feature array |
| FREEPOINT package `REFE/raw/2026-10-17-free-point-navtest/` and `C:/Users/Admin/freept/` | picks / points (`fp_picks.py`), pod scripts | **no** (`find` -> 0 hits in `freept`; the package's `fp_picks.py` is a seam writer) |
| SR-TTC S0: `srttc_s0_dump.py` (writes `--feat-out`), `srttc_s0_plan.py`, `S0_HANDOFF.md`, run tree `C:/Users/Admin/srttc_s0/` | features of the navtrain **fit sets** (S0 bank, running now, not opened) | **no** -- different tokens (navtrain), not navtest |
| `C:/Users/Admin/srttc_s0/work/dgdump/dgdump_feat.npz` (DG-DUMP) | `prehead_features float16 (4,64,256)`, 4 tokens of one navtest log | **4 of 12,146 tokens (0.03 %)** -- a proof of the forward, not a bank |
| `D:/Projects/TanitAD-artifacts/p2_heldout_table*/prehead_features_p2_heldout.npz` | 1,705 held-out **navtrain** tokens (the EP-R1 DEV set) | no (not navtest) |

MEASURED: the `find` outputs and header listings above were run 2026-10-19; the real-file input check is `raw/refe_head_confirm_inputs_check.json` (no outcome member of the table opened).

## 3. The forward, exactly

| item | value |
|---|---|
| script | `REFE/raw/2026-10-18-sel-rank/code/srttc_s0_dump.py`, fan path: `--ckpt <P2> --plan plan_cNN.json --out cNN_dump.npz --feat-out cNN_feat.npz --device cuda` (**no `--lean`**: lean drops the features and the fan; **no `--sets-dir`**: that is the S0 fit-set path). Run with the wrapper shape of `srttc_s0_dgdump.py`: `cwd` = the code dir, `env = EC.env_driverl()` from `eval_checkpoint.py` (`P2HO_TREE`), python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` |
| plans | `make_navtest_bank_plan.py` (new, CPU, tested): reads the landed `chunks/chunks.json` + `cNN_tokens.json` of `2026-10-16-p2-full-table` (sha256 verified), the table's `token`/`log` only, checks the 14 chunks partition the 12,146 tokens once and that all 136 nuPlan DBs exist, writes `plan_cNN.json` in the `--plan` shape of the real DG-DUMP plan and `bank_commands.txt` |
| checkpoint | `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` (the one the full table and the DG-DUMP used) |
| device | **GPU** (ViT-L/16 image forward per token, ~1.6 s/token). CPU is not viable: ESTIMATED, the landed CPU smokes ran 4-8 tokens; at ViT-L CPU speed 12,146 tokens is days, and CPU-vs-GPU noise (1.74e-3 on logits, INHERITED: the comment in `srttc_s0_dgdump.py`) would put C4's bit-identity out of reach |
| dev-box GPU hours | **5.43 h** = sum of the 14 landed per-chunk dump `seconds` (19,531 s over 12,146 tokens, 0.62 tok/s; per chunk 0.46-0.73 tok/s, the slow chunks c05-c07 overlap the scoring workers). MEASURED for the landed forward, `p2-full-table/state/chunk_NN.json`; **INHERITED for this job** (same planner, same frames resolver, plus one forward hook on `score_head`'s input, which the DG-DUMP showed is free: 42.6 s for 4 tokens including model load). Plan 6 h with margin |
| GPU lock | queue behind S0 with `gpu_lock.py acquire` (never remove another job's lock) |
| disk | features `[12146,64,256]` fp16 = **398,000,128 B**; the fan dump beside it (proposals + proposals20 + logits) ~0.28 GB (ESTIMATED: 14 landed `dump_main` chunks sum to ~92-98 MB without `proposals20`, plus 12,146 x 15,360 B for it). Total ~0.7 GB; D: has 120.8 GB free |
| RAM at apply time | `refe_head_confirm.py` streams chunk by chunk (<= 993 tokens): working set a few hundred MB. A whole-table float64 pick would be ~1 GB of temporaries; the module chunks it (found while scoping, fixed) |
| acceptance | per chunk: `cNN_feat.npz` + `cNN_dump.npz`, all 12,146 tokens once; then `refe_head_confirm.py manifest` (sha256 of every npz) and `picks`, whose controls decide: P2's head reproduces the stored logits from the features (<= 2e-3), C0 identity, C4 dump proposals == the table's with max abs **0.0**, dump pick == table pick, logits <= 1e-4 |

## 4. What is already verified on the REAL files (no outcome opened, no GPU)

`raw/refe_head_confirm_inputs_check.json` (MEASURED, 22.5 s CPU): the landed table hashes to `33441082...` and loads under the pinned shape (12,146 / 136) **through this module's guard** (the dev guard would
refuse the names `table_p2_full` and `navtest`); the planner rule on the stored logits reproduces the stored pick on **12,146 / 12,146**; the banked P2 seam `stage2_p2_full.npz` equals the table's executed pose of
P2's pick with **max abs 0.0 m**, fingerprints equal, so the applier's banked-seam control is GREEN on the real data; the 14 chunks partition the table once, every token file's sha256 and token_log verified, all 136 DBs exist.

## 5. Risks and decisions this note hands to the Master Mind

1. **Schedule.** 5.43 GPU-h after S0. The bank is outcome-free and spends no look, so building it does not wait for a dev gate; the registered "S3 only on a dev pass" is a cost rule. The bank serves all three readings, so one forward covers SR-TTC, EP-R1 and the joint. **Decision: queue it behind S0 now, or only after a dev pass.**
2. **C4 on the 4 known frame-control logs (HYPOTHESIS).** The landed full-table run carried 36 known over-bar frame-control tokens in 4 logs (max 6.19 m, `ZZSEAM_FAIL` lines for c01-c03). The DG-DUMP reproduced the table bit-identically on 4 tokens of log 1 only. If the new dump differs from the table on tokens of those logs, C4 reads RED and the applier refuses (no waiver is registered). The cheap pre-check is the DG-DUMP on 4 tokens of one of those logs.
3. **Partial bank.** A look on fewer than the 12,146 tokens breaks parity with every landed navtest number; the applier refuses an incomplete bank.
4. **`--lean` trap.** The S0 launcher pattern uses `--lean`; used here it yields no features. `make_navtest_bank_plan.py` never emits it and its test asserts that.
