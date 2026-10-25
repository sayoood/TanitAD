# HANDOFF: AR-3 for PREREG_V8F_LORA, arm L seed 0, launch-gate dry run (dev box, CPU, no GPU/box 2/A40)

**Status 2026-10-26 ~00:40.** Priorities 1 and 2 are DONE, 3 and 4 are RUNNING detached. Next agent: run `C:/Users/Admin/venvs/tanitad/Scripts/python.exe <this dir>/ar3_read.py`; it prints every pending result. Do not wait alive on them.

## Tip
Worked on **c2fad328f517899dc8808c054d096824d96f66b2** (`git ls-remote` at start; code exported with `-c core.autocrlf=false`: plain `git archive` CRLF-converts here and the gate rejected the tree, 2,626 files differ).
The remote moved to **15abe8d96eae662f720678c68fe3918fe9f9f759** during the run (5 commits). Diff vs mine under stack/, taniteval/, tools/ and the v8f-mini code dir: `v8f_mini.py` +104 (HACC post-map, inference-only, optional buffer absent by default), new `v8f_mini_eval_speedcal.py`, 2 new tests, speedcal code. Trainer, LoRA encoder and PREREG_V8F_LORA are unchanged. Every result must be re-run on the tip it is bound to before any token.

## Headline finding
**AR-3 as written cannot be satisfied by arm L.** Plan 13.5 says `PROFILES["v8f"]`, which is the refcv8-chassis rung-alpha profile (trainer `refc_v3_train.py`, `launch_gate.py:864-1027`). Arm L runs `v8f_mini_train.py`.
* MEASURED: `launch_gate.py run --profile v8f` on L's argv gives **FAIL, 73 refusals** (missing `--refcv8`, `--residual-prior`, ...). G-HYG and G-DVB end ERROR. `raw/gate_v8f_profile_L.log`, `raw/gate_v8f_profile_L/`.
* MEASURED: that trainer's parser rejects the argv on the first flag, `--arm: invalid choice: 'M2b' (choose from hier, flat)`. `raw/refc_parser_on_L_argv.txt`.
* `launch_gate.py` has no v8f-mini profile and 0 "lora" strings. `grep v8f_mini_train stack/ops` = 0 hits; the only token-verifying supervisor is `sup_refcv7.sh`. So **no profile, token or gated exec exists for any v8f-mini arm**. Every earlier arm (WC, PGT, GRID, HIST, DSCALE) used the trainer's in-process gate: `declared_vs_built` + `live_check` + probes, written to `selfcheck.json`.
* PREREG item 6 ("weights sha256 bound in the token") has no mechanism: the weights load from the HF cache, not an argv path, so `data_manifest` cannot bind them. The in-trainer equivalent is `bad_encoder` (`v8f_mini_train.py:246`), refusing a sha that is not the builder's `dcb2e451...8179`.
* Also: on Windows `--cpu-only` cannot pass. `launch_gate.py:7303` sets `CUDA_VISIBLE_DEVICES=""` and Windows deletes empty variables ("CUDA is visible"). Irrelevant until a profile exists.

## Arm L seed 0 argv (`raw/argv_L_s0_canonical.json`; trainer argv_sha `e9cf6e373a32315a`, gate sha256 `1df0cf38...b7288`)
* **B part**: verbatim what the queue prints for `train M2b 0` (`raw/queue_dryrun_lines.txt`, from `v8f_mini_box2_queue.cmd:92`, `A=`): `--arm M2b --seed 0 --grid pool4x10 --cache-dir D:\tanitad\cache_full --v9-train ...v9_labels_train.npz --v9-train-md5 83e30bd3... --lead-train ...lead_train4572.npz --speed-train ...refcv6_speed_max_v8_train.jsonl --join-train ...join_train.npz --p10 ...p10_prior.pt --kin-mode yaw --tracks ...native_tracks.pt --exclude 540:600 --batch 64 --steps-mult 4 --device cuda`.
* **LoRA part**, from PREREG "The arm": `--encoder lora --lora-r 32 --lora-alpha 32 --lora-targets q,v --enc-ckpt --accum 64`. No `--amp` (Amendment 2). The values are defaults but are named, so the argv carries them.
* **Not committed anywhere, so placeholders**: `--out`, `--frames-root <VENUE>`, and `--accum` (any divisor of 64). The token binds the exact argv, so these must be fixed first. There is no `trainlora` queue stage: only `identitylora` and `smokelora`.

## Checks
| check | verdict | evidence / reason |
|---|---|---|
| **G-HYG** | **FAIL / not implemented** | v8f profile: ERROR (above). v8f-mini: `V8FMiniConfig` is a plain `@dataclass` (`v8f_mini.py:140`), no slots/`__setattr__` guard; the trainer builds it from explicit fields. |
| **G-DVB** | **FAIL on exact shape (frames only)**; coverage PASS | Static: 66/66 parser flags have a `FLAG_CHECKS` entry, none missing (`raw/static_gdvb_coverage.txt`). Exact levers, no `--max-clips`, CPU: `ZZDVB-REFUSEDZZ ... 3972 train clips have no <clip>.v2ep.pt`, the **only** item in the list (`raw/selfcheck_L_fullshape.log`). So trainable count 3,145,728, sha, r/alpha/targets, optimiser group, B=0 all matched. The exact argv cannot pass on any host until frames for all 4,512 train clips exist (`v8f_mini_train.py:248`); box 2 deleted 3,972 (Amendment 2). That is AR-4. |
| **G-LIVE** | **RUNNING** (pid 53544) | `raw/selfcheck_L_live4.log`, `ar3_work/runs/L_live4/selfcheck.json`. AR-2's shape (`--max-clips 4`, the 4 declared clips' frames) on the tip trainer, CPU bf16. PASS = `declared_vs_built` PASS, `live.pass` true, A grad at step 0 exactly 0.0, B > 0, frozen DINOv3 grad exactly 0.0, no missing term. Dev-box inputs are NOT production bytes: pooled tokens for the 4 clips come from `test_runscript` (md5 differs from box 2's manifest), the join is rebuilt locally from `box2_ship/v9ship` (md5 83e30bd3 verified). The identity probe is informational (CPU vs GPU, about 0.03 rel-L2 expected). |
| **G-CLOCK** | **NOT RUN / no gate** | No label-clock check exists for v8f-mini. The only real-window clock test, `test_v8f_mini.py::test_five_hand_checked_real_windows`, opens a file under `C:/Users/Admin/qland` and is deselected (brief). |
| **G-EVAL** | **PENDING (P1 tests)** + NOT RUN (real) | Mechanism tests `test_v8f_lora.py::test_eval_route_reproduces...`, `::test_real_loader_never_moves...`. Real epsilon (banked M2b through the merged-encoder eval cache, max abs diff <= 1e-3 m) needs box 2's GPU and `v8f_lora_eval_cache.py`: NOT RUN. |
| **G-CKPT** | **PENDING (P1 tests)** | `test_lora_run_resumes_as_the_same_run`, `test_lora_state_roundtrip_and_refuses_a_different_shape`. |
| **G-SUITE** | **PENDING** (driver2) | Plain pytest on the clean tree `C:/Users/Admin/ar3_gt` (git archive of c2fad328), suites stack, taniteval, tools. NOT the gate's job: no baseline compare, no mutation audit, 1 test deselected (qland). |

AR-2 run 3's `selfcheck.json` (box 2, tip c0769eaa, argv sha c37dec91cacee) shows G-DVB and G-LIVE PASS. It is **not evidence for the tip**: since then `v8f_mini_train.py` +263 and `v8f_mini.py` +85 lines (acc/o_interface/code_mix probes). live4 is the first real-weights run of the tip trainer through `--encoder lora`; tests use stubs.

## LoRA levers vs G-DVB
No flag lacks an entry. Two weak entries:
1. `accum` is "enc:accum" but only divisibility is checked (`v8f_mini_train.py:228`). `built_levers()` has no accum; the executed micro-batch is not read back from the built path. Smallest fix: add `accum` to `DinoV3LoRAEncoder.built_levers()`/`bad_encoder`, or assert `chunk == (batch/accum)*hmax` where `cached_grad_step` is called.
2. `amp` is kind "run" and not refused under `--encoder lora`, though Amendment 2 pins it OFF. Smallest fix: a `bad_encoder` line `if a.amp: bad.append(...)`.
Non-flag levers (dropout 0, fp8 STE, grid, checkpointing) are asserted at `:224-225`.

## Pending jobs (mine; scripts in `raw/`)
* `ar3_driver2.sh` (log `raw/driver.log`) waits for live4 (RAM was 3 GB free), then P1 (`raw/pytest_lora_subset.{log,xml}`), then G-SUITE (`raw/gsuite_{stack,taniteval,tools}.{log,xml}`), each pytest only when <= 2 other python.exe pytest lines. Ends with `ZZDRIVER-DONE`. C: has ~6 GB free; pytest temps are under `ar3_work/bt1`, `bt2`.

## Next step
1. Run `ar3_read.py`; record G-LIVE, P1, G-SUITE in the table.
2. **Decision for the PI/MM (blocks AR-3)**: (a) redefine AR-3 for track M as "in-trainer gate PASS (`selfcheck.json`) on the exact commit and argv + P1 tests green + G-SUITE", a one-paragraph PREREG amendment; or (b) build a `v8f_mini` profile in `launch_gate.py` plus a supervisor and a weights-sha `gate:` input (a build task). The MM can do (a) now.
3. Then fix `--out`, `--frames-root`, `--accum`; re-run on the then-current tip with all 4,512 frames (AR-4); add the two G-DVB fixes first.

## Manifest (all under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/ar3/`)
`HANDOFF.md`, `ar3_read.py`, `raw/` (argv_*.json, queue_dryrun_lines.txt, gate_v8f_profile_L.log + dir, refc_parser_on_L_argv.txt, static_gdvb_coverage.txt, selfcheck_L_fullshape.log, selfcheck_L_live4.log, driver scripts, tip.txt). Off-scratchpad, mine: `C:/Users/Admin/ar3_work` (tip export, dev cache `dc/`, `runs/L_live4`), `C:/Users/Admin/ar3_gt` (clean tree), `ar3_suite`, `ar3_gatekey` (gate HMAC key). Nothing staged, committed or pushed; box 2, qland and box2_ship untouched (read-only: box2_ship logs, v9ship, frames).
