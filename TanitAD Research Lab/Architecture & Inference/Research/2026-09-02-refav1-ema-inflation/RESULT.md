# RESULT — E-ARCH-TSC-2: on the dev-box rig the live recipe did NOT reproduce the tactical-target inflation, so R1 is VOID and E was not run; the OFF checkpoint REFUSES to resume under `--ema-targets`

`E-ARCH-TSC-2` · 2026-09-02 · Architecture & Inference FlyWheel · dev-box RTX 4060 (Thor and every pod:
**not contacted**) · **Tier T0** — mechanism probe on 20 EVAL clips used as training data; **every model here
is DISCARDED and no number from this rig may enter `MODEL_REGISTRY.md`** (SPEC §5, verbatim in §6).
SPEC: `SPEC.md` in this package — pre-registered by the Master Mind before any arm ran; arms, order and
reads R1–R7 were executed as committed.

## Verdict (headline)

1. ⛔ **R1 VOID — the slice rig is insensitive to the inflation.** Arm **B′** (the live recipe, the
   deliberate-regression arm that *must* inflate) ran 250 steps and `tgt_std_tac` went **0.02006 → 0.01968
   (×0.98)**; its largest value on any of the 26 rows was ×2.25 (step 220). The committed VOID branch is
   "< 3". ⇒ per SPEC §2 and the brief, **arm E was NOT run: it is uninterpretable on this rig.**
   **Whether the EMA teacher pins the scale is therefore UNANSWERED here** — not refuted, not confirmed.
2. ✅ **R7 — an OFF checkpoint does NOT resume under `--ema-targets`.** `refa_v1_train.py --resume
   --ema-targets` on B′'s `ckpt.pt` dies in 2.8 s with `RuntimeError: Error(s) in loading state_dict for
   RefAV1: Missing key(s) in state_dict: "ema.tac_queries", "ema.adapter.pos", … "ema.str_read.1.bias"`
   (**20 `ema.*` keys**). The trainer calls `model.load_state_dict(ck["model"])` strictly
   (`refa_v1_train.py:348`) and nothing initialises the teacher from the student. ⇒ **option (b) of
   C-REFAV1-TAC-INFLATION ("switch the live run at a checkpoint") is a CODE CHANGE, not a 5-minute
   operation** — a non-strict load plus an explicit teacher←student copy at the switch.
3. ✅ **R4 CONFIRMED — but read at step 220, TRUNCATED.** Arm **A′** (adapter space, no detach = the
   original defect) reproduces the collapse: `adapter_std` **0.4508 → 0.3349 (−25.7 %)** at step 220,
   participation 27.3 → 9.0 (heading for the 8.56 floor), the adapter-space target std 0.454 → 0.327 while
   the loss fell 0.201 → 0.085 — the collapse minimum, as TSC-1's arm A on Thor (−35.7 % at 250).
   ⚠️ The A′ process was **killed at ≈ step 226 by the tool harness's ~60-min background-task limit**
   (B′ at 55.7 min survived; A′ at 61 min did not — §4b), so R4 is read at the last banked row, not the
   committed 250, and no checkpoint exists (first save was due at 250). The committed −15 % threshold was
   already passed at step 110 (−21 %) and every row from 110 on sits below it.
   **⇒ the rig sees the COLLAPSE direction and is blind to the INFLATION direction: the VOID is specific
   to inflation, not a dead instrument.**
4. ✅ **R5 (known value) CONFIRMED** on B′: `tgt_std_op` 0.9825 first row, 0.9794 last, mean of first five
   rows 0.981 vs last five 0.976 (drift −0.5 %); single rows range 0.923–1.052 (bs 1 = one window per row).
5. ⚠️ **Nothing on this box fits in 8 GB** — bs 8 (Thor's) and bs 4 OOM at the WDDM over-commit ceiling
   (~22.5 GiB "allocated" on an 8 GiB card), bs 2 runs at ≈ 41 s/step, **bs 1 ran every arm at 13.4–15.8
   s/step with `torch.cuda.max_memory_allocated()` = 12.29 GB (B′), i.e. paging through host RAM.** The arms
   compare to each other, not to Thor (SPEC §2); and **bs 1 is the prime suspect for the VOID** (§4a).

**What the Master Mind now decides:** (i) re-run B′/E where bs 8 fits (Thor between checkpoints, or an
A40/A6000 pod — ~90 min per 250-step arm at Thor's 20.7 s/step) — the only rig that has shown the
inflation is a bs-8 rig; (ii) treat the live-run switch (b) as a code change (non-strict resume + teacher
init) that needs its own test before it touches Thor; (iii) whether to accept the 250-step bs-1 rig at all
for future TSC probes — it reproduces the **collapse** (R4) but not the **inflation** (R1).

## 1. What actually ran (the rig)

| item | value | class |
|---|---|---|
| code | repo HEAD `bdd1617` (`agent/arch-inf-20260803`). Private mirror `C:\Users\Admin\tsc2\stack` = robocopy of the dev-box mirror `C:\Users\Admin\tanitad-wt\stack` **with the HEAD BLOB of `refa_v1.py` written over it**, because the worktree/mirror copy carries the in-progress Task-C edit (blob `0fe75837`, +207/−12 lines, `plan()`-side only — `canonical_controls`, the goal-as-kinematics table; **not run here**). Blobs that ran, recorded by `git hash-object` in every arm's `config.json`: **`refa_v1.py` `8264ded883d5ade15105f8e54fe9e5fe51621ad5` = `HEAD:stack/tanitad/refs/refa_v1.py`**, `refa_v1_train.py` `d94a163238f6788f62ba669a9cc035f5262bfd72` (= HEAD), `refav1_loader.py` `5b34f495fd0b2ade656a5ee731d1f1cb65712aca` (= HEAD). The launcher (`raw/tools/run_arm.py`) **asserts** the imported `tanitad` is the mirror's, not the G: editable install (`tanitad_file` in each `config.json`). | MEASURED |
| trainer | `stack/scripts/refa_v1_train.py::main`, called in-process, **unmodified** — read-only on all code, per brief | MEASURED |
| data | the 20 v7.2 **EVAL** clips pulled from Thor (`C:\Users\Admin\refav1_eval_slice\{fp8,eps}`; the 20 ids are in `raw/cache_index.json`). Features `float8_e4m3fn [100–101, 640, 1024]` (DINOv3 ViT-L/16, 0.2 s grid), **hard-linked** into `C:\Users\Admin\tsc2\cache` with an `index.json` carrying the slice index's geometry verbatim (`n_tokens 640, d_enc 1024, hfov_deg 120.0`; `verify_cache` PASS). v2eps 201 frames; `clip_id` == file stem on 20/20; the loader's grid check `T_c == ceil(T_ep/2)` PASS on 20/20 | MEASURED (`raw/tools/inspect_and_build_cache.py`) |
| labels / nav | `s2_labels_v7.2_eval.jsonl.gz` (147 records, md5 `aa12c948f062181c3297265b51526ec5`) for BOTH `--labels` and `--nav`. The trainer's own join report, identical in every arm (`raw/*.log`): **`labels 20/20 eps (0 missing), 199/739 windows in-band +-[2.0]s | nav 20/20 eps (0 missing)`**, `allow_oracle_nav=True`. **No override was needed**: `refa_v1_train.py` passes no `labels_expected_records`, so a 147-record blob over 20 episodes is not refused. `forward()` consumes `lat_label, lon_label, nav_cmd, route_label, str_ext_actions, str_ext_targets, v0`; the loader's `nav_valid` mask is dropped (a validity mask, not a label — the trainer prints this) | MEASURED |
| windows | 739 windows over 20 episodes; `--seed 0` ⇒ **identical window order in every arm** (B′ and A′ share their step-1 `tgt_std_tac` to 5 digits: 0.02006 / 0.02006) | MEASURED |
| box | RTX 4060 **8 GB** (`8188 MiB`; the brief's "8.6 GB" is this card in decimal GB), torch `2.11.0+cu128`, Python 3.13.5, Windows 11, venv `C:\Users\Admin\venvs\tanitad`. No other python process on the GPU at any launch (`gpu_compute_apps_before` in each `config.json`) | MEASURED |

## 2. Batch size — NOTHING fits in 8 GB; bs 1 is the largest batch that RUNS

MEASURED with the unmodified trainer, 2 steps each (`raw/fit_probe/`; `config.json` carries
`torch.cuda.max_memory_allocated()`, the only admissible device-memory number):

| bs | outcome | `cuda_max_memory_allocated` | step time |
|---|---|---|---|
| 8 (Thor's live batch) | ⛔ `OutOfMemoryError` — *"22.47 GiB is allocated by PyTorch … GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free"* | 24.12 GB | — |
| 4 | ⛔ OOM at 22.48 GiB allocated | 24.14 GB | — |
| 2 | ran (2 steps) | **23.12 GB** | ≈ 41 s/step (`elapsed_s` 54.3 → 95.1) |
| **1** | ran (2 steps) | **12.29 GB** | 13.1 s/step (11.7 → 24.8) — **used for every arm** |

⚠️ 12.29 GB "allocated" on an 8 GB card means the run lives in **WDDM shared host memory** — the trap the
step-profile package recorded (`…/2026-09-02-refav1-step-profile/RESULT.md` §1–2: capped at 6.4 GB the same
card OOMs at *every* batch; uncapped, PyTorch over-commits to ~22.5 GiB and then fails). This is an
**independent reproduction of that finding with the live trainer**: the 30-step operative rollout keeps all
30 steps' activations for backward (`--bptt-truncate` bounds the gradient path, not the stored graph),
≈ 7.5 GB per batch row at K=30, plus 2.8 GB of weights/grads/AdamW. Also MEASURED here:
`torch.cuda.set_per_process_memory_fraction(1.0)` does **not** cap (22.5 GiB went past it); the profiler's
0.80 does. ⇒ the arms' **s/step (13.4–14.2 s at bs 1) is a paging number and compares to nothing off
this box**; the *instruments* (losses, `tgt_std_*`, `adapter_std`, `participation`) do not care where the
memory lives. **bs 1 ≠ Thor's bs 8: the arms compare to each other, not to the live run** (SPEC §2).

## 3. Command lines and timing (the launcher appends `--out`; every flag is the trainer's own)

Common: `--cache C:\Users\Admin\tsc2\cache --episodes C:\Users\Admin\refav1_eval_slice\eps --labels <eval blob> --nav <eval blob> --steps 250 --bs 1 --lru 16 --log-every 10 --save-every 250 --seed 0 --min-participation 0 --device cuda` (lr 3e-4, adapter ×0.1, clip 1.0, wd 0.01 = trainer defaults; `sanity()` PASS).

| arm | added flags | run dir (dev box) | raw copy here | s/step (rows 10→250) | wall | `max_memory_allocated` |
|---|---|---|---|---|---|---|
| **B′** (live recipe = deliberate regression) | `--target-space frozen --detach-aux-targets --bptt-truncate 15` | `C:\Users\Admin\tsc2\Bp` | `raw/B_prime/{train_log.jsonl,config.json}`, `raw/B_prime.log` | **13.35 s** | 3,321 s | 12.29 GB |
| **R7** (resume) | copy of `Bp` + `--resume --ema-targets --steps 260` (B′'s other flags) | `C:\Users\Admin\tsc2\R7_resume` | `raw/R7_resume/{config.json,config_Bprime_source.json,robocopy.log,train_log_inherited_from_Bprime.jsonl}`, `raw/R7_resume.log` | — (died at load) | 2.8 s | 2.83 GB |
| **E** (B′ + `--ema-targets`) | **NOT RUN** — STOP rule (R1 VOID) | — | — | — | — | — |
| **A′** (adapter space, the ORIGINAL defect) | `--target-space adapter --no-detach-aux-targets --bptt-truncate 15` | `C:\Users\Admin\tsc2\Ap` | `raw/A_prime/{train_log.jsonl,config.json}`, `raw/A_prime.log` (stdout, buffered — ends at step 140; the jsonl is flushed per row and is the primary source) | **15.83 s** (rows 10→220) | **killed at ≈ 61 min** (last row step 220 at 23:11:50) | UNVERIFIED — killed before the launcher wrote it (ESTIMATED ≈ B′'s 12.29 GB) |

Execution order: fit probe (21:13 Berlin) → **B′** (21:17–22:12) → **R7** (22:13, needs only B′'s
checkpoint; slotted here because it costs seconds) → R1 read (VOID) → **A′** (22:14 – killed 23:15 at ≈ step 226). The
SPEC's arm order B′ → E → A′ is unchanged; E is absent by the SPEC's own rule. After ~20 steps of B′ the
measured 12.5 s/step projected ≈ 52 min per arm, ≈ 2.8 h for three arms + R7 — inside the GPU window
that closed at 00:17 Berlin (a cron-launched checkpoint read on this GPU).

## 4. The reads, against the COMMITTED criteria (SPEC §3) — every number from the raw row at the named step

| # | read (SPEC) | CONFIRMED if | REFUTED / VOID if | MEASURED | verdict |
|---|---|---|---|---|---|
| **R1** | B′ `tgt_std_tac(250)/tgt_std_tac(1)` | > 10 | < 3 ⇒ VOID | **0.02006 → 0.01968 = ×0.98**; max over rows ×2.25 (0.04523 @ 220); `tgt_std_str` ×0.84 (0.03885 → 0.03254) | ⛔ **VOID** — the rig does not reproduce the live inflation |
| **R2** | E `tgt_std_tac(250)/tgt_std_tac(1)` | ≤ 2 | > 5 | — | **VOID — E not run** (STOP rule) |
| **R3** | E tactical loss share at 250 | < 10 % | ≥ 25 % | — (B′'s own share for reference: 0.06 % @ 1 → 0.99 % @ 250, max 9.8 % on any row @ 20) | **VOID — E not run** |
| **R4** | A′ `adapter_std` | falls > 15 % | flat / rising ⇒ rig insensitive to collapse | 0.4508 → **0.3349 @ step 220 = −25.7 %** (last-3-row mean 0.342 = −24.2 %; 11/22 row-to-row deltas down, the trend monotone through 0.45 / 0.43 / 0.41 / 0.39 / 0.36 / 0.35 / 0.33); participation 27.3 → 9.0; adapter-space target std 0.454 → 0.327; `tgt_std_tac` ×0.61, `tgt_std_str` ×0.41; loss 0.201 → 0.085 while the targets shrank | ✅ **CONFIRMED — read at step 220 (TRUNCATED: killed ≈ step 226, §4b)** |
| **R5** | `tgt_std_op` in B′ (and E) | ≈ 1.0 throughout | drifts ⇒ instrument fault, VOID | B′: first 0.9825, last 0.9794, mean 0.981; first-5 mean 0.981 vs last-5 0.976 (**−0.5 %**); rows 0.923–1.052 | ✅ **CONFIRMED** (no drift; per-row spread is bs-1 scene noise) |
| **R6** | E `loss_feat_op(250)` vs B′ | within 10 % | > 25 % worse | — (B′: 1.4103 → 0.5761; last-5 mean 0.585) | **VOID — E not run** |
| **R7** | OFF ckpt under `--ema-targets` | loads, EMA initialised from student | refuses on missing keys | `RuntimeError … Missing key(s) in state_dict:` **20 `ema.*` keys** (§5) | ⛔ **REFUSES** |

Operationalisations stated post-hoc (the SPEC gave none): R5 "drifts" = |last-5 mean / first-5 mean − 1| > 5 %;
R4 "flat" = change ≥ −2 %. All computed by `raw/tools/reads.py` → `raw/reads.json` (the same numbers).

### 4a. Why R1 is VOID here and not on Thor — the rig differs from the live run in ONE dominant way

| | live run / TSC-1 arm B (Thor) | this rig (B′) |
|---|---|---|
| batch | **8** | **1** (nothing larger fits, §2) |
| windows seen by step 250 | 2,000 | 250 |
| data | Thor corpus (TSC-1: "identical data", `--lru 64`) | 20 EVAL clips, 739 windows |
| `bptt_truncate` | live run 15 (TSC-1 arms: 0) | 15 |
| `tgt_std_tac` 1 → 250 | TSC-1 B: **0.0571 → 0.8418 (×14.7)**, ×2 by step 125, ×7 by 200 (`…/2026-09-02-refav1-target-space-collapse/raw/TSC-B-frozen.jsonl`, MEASURED from the banked raw); live run ×20 by 250, ×104 by 500 (SPEC §1) | **×0.98**, never above ×2.25 |
| `adapter_std` 1 → 250 | TSC-1 B: 0.4778 → 0.5937 (+24.3 %) | 0.4557 → 0.5782 (**+26.9 %**) — the same healthy rise |
| `loss_feat_tac` 1 → 250 | TSC-1 B: 0.0022 → 0.023 (×10.5) | 0.0018 → 0.0211 (×11.6) |

The adapter grows the same way, the tactical **loss** grows the same way, but the tactical **target**
does not move. The target `tq = tac_pool(tac_queries, adapter(std(f_future)))` inflates only if
`tac_pool`/`tac_queries` change scale, and those receive gradient solely through the *prediction* side
(the target is detached) — one window of gradient per step here against eight on Thor. ⇒ **HYPOTHESIS
(not measured): the inflation is a per-window-count effect that bs 1 slows below the 250-step horizon,
not a data effect.** The cheapest discriminating experiment is a bs-8 B′ on Thor between checkpoints
(~90 min) or a 1,000-step bs-1 B′ here (~3.7 h); the SPEC's E then follows on whichever rig inflates.
⚠️ `tgt_std_tac` at bs 1 is a per-channel std over one window's 10 tactical steps × 64 queries, so its
absolute level (0.020 vs Thor's 0.057 at step 1) is not comparable across batch sizes; only the
within-arm ratio is, which is how R1 is defined.

## 5. R7 — resume tolerance (SPEC §4), the exact evidence

`raw/R7_resume.log` (B′'s `ckpt.pt` copied by robocopy, md5 `65245a78b5c8591b8db2ab9132598459` on both
sides; `raw/R7_resume/robocopy.log`), then `refa_v1_train.py … --resume --ema-targets --steps 260`:

```
[run_arm] R7: EXC RuntimeError: Error(s) in loading state_dict for RefAV1:
	Missing key(s) in state_dict: "ema.tac_queries", "ema.adapter.pos", "ema.adapter.proj.0.weight",
	"ema.adapter.proj.0.bias", "ema.adapter.proj.1.weight", "ema.adapter.proj.1.bias",
	"ema.adapter.proj.3.weight", "ema.adapter.proj.3.bias", "ema.adapter.tmix.weight", "ema.adapter.tmix.bias",
	"ema.adapter.out.weight", "ema.adapter.out.bias", "ema.tac_pool.in_proj_weight", "ema.tac_pool.in_proj_bias",
	"ema.tac_pool.out_proj.weight", "ema.tac_pool.out_proj.bias", "ema.str_read.0.weight", "ema.str_read.0.bias",
	"ema.str_read.1.weight", "ema.str_read.1.bias".  ; wall 2.8s; cuda_max_memory_allocated 2.83296256 GB
```

Reading, from source (HEAD blob): `_EmaTargetPath` is built in `RefAV1.__init__` when `ema_targets=True`
and its parameters live in the state_dict under `ema.*` (`refa_v1.py:766-831, :978`; pinned by
`tests/test_refa_v1_ema_targets.py::test_the_teacher_lives_in_the_state_dict_when_on`); the trainer's
resume is `model.load_state_dict(ck["model"])` with the default `strict=True` (`refa_v1_train.py:348`).
There is no path that initialises a missing teacher from the student. **Nothing was fixed** (brief:
report only). What a fix would need, for the Master Mind: a resume that loads with `strict=False`, then
verifies the ONLY missing keys are `ema.*`, then copies `pairs()` student→teacher — and a test that an OFF
checkpoint resumed under EMA reads `tgt_std_tac` identical to the student's on the first forward.

## 6. What this does NOT settle (SPEC §5, verbatim)

* Nothing here is a capability claim. Tier **T0**, mechanism only, discarded models, 20 EVAL clips used as
  training data (admissible ONLY because no model from this rig is ever evaluated or registered).
* Whether inflation HURTS the final model — that is what the live run's operative term and the step-1000 /
  later T1 reads answer. This SPEC answers only whether the EMA path pins the scale.

Rig caveats (mine, not the SPEC's): **bs 1, not 8**, over-committed memory (§2), one window per logged
row (per-row noise is larger than Thor's bs 8; the committed reads are still read at the committed step,
5-row means appear only as supplementary); every checkpoint here is from the **eval** slice — kept only
under `C:\Users\Admin\tsc2\{Bp,Ap}\ckpt.pt` for a possible R7 re-test, never to be evaluated, and to be
deleted once the Master Mind has read this.

### 4b. A′ was truncated at step 220 — what happened, and what it does and does not change

MEASURED: the A′ launcher ran as a background Bash task of the agent harness; the harness reported the
task **"killed"** at 23:15:07 Berlin; immediately after, no python process held the GPU (711 MiB used, the
desktop), `Ap/train_log.jsonl` ends at step 220 (written 23:11:50, ≈ 15.8 s/step ⇒ the kill landed near step
226), `Ap/config.json` carries no final status (the launcher's `finally` never ran — a hard kill), and there
is no `ckpt.pt` (`--save-every 250`). B′ ran 55.7 min under the identical mechanism and completed; A′ was
at 61 min. ⇒ **the harness's background task has a ~60-min ceiling — a TRAP for any run over ~55 min:
launch such runs detached (`Start-Process` / `nohup`-equivalent), not as a tool background task.** The
same ceiling would have killed E (≈ 56–66 min at this s/step).

What it changes: R4 is read at step 220 instead of the committed 250 — a **deviation, stated**. What it
does not change: the read's direction and magnitude. The criterion is a fall > 15 %; A′ passed −15 % at
step 110 and never returned above it (rows 110–220: −21 %, −20 %, −17 %, −22 %, −18 %, −23 %, −21 %, −20 %,
−23 %, −23 %, −24 %, −26 %). A′ was not re-run because a 66-min re-run could not finish before the
00:17 Berlin GPU hand-over (the cron-launched checkpoint read). **Re-run when the GPU is free** — one
command, ≈ 66 min, run DETACHED:
`python C:\Users\Admin\tsc2\tools\run_arm.py --arm A_prime --out C:\Users\Admin\tsc2\Ap2 -- <common flags, §3> --target-space adapter --no-detach-aux-targets --bptt-truncate 15`
(the `raw/A_prime/config.json` carries the exact argv). A′'s stdout log (`raw/A_prime.log`) stops at step
140 because the kill dropped the buffered redirect; the jsonl is flushed per row and is complete to 220.

## Deliverable manifest

| artifact | location | copies |
|---|---|---|
| this result | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-02-refav1-ema-inflation/RESULT.md` (+ `C:\Users\Admin\tsc2\RESULT.md`) | 2 |
| SPEC (unchanged) | `repo:…/2026-09-02-refav1-ema-inflation/SPEC.md` | already tracked |
| B′ raw log + provenance | `repo:…/raw/B_prime/{train_log.jsonl,config.json}`, `raw/B_prime.log`; dev box `C:\Users\Admin\tsc2\Bp\` (+ `ckpt.pt` 2.09 GB, **dev box only, to be deleted**) | 2 (ckpt: 1) |
| A′ raw log + provenance (**TRUNCATED: rows 1–220, no checkpoint**) | `repo:…/raw/A_prime/{train_log.jsonl,config.json}`, `raw/A_prime.log`; dev box `C:\Users\Admin\tsc2\Ap\` | 2 |
| R7 evidence | `repo:…/raw/R7_resume.log`, `raw/R7_resume/{config.json,config_Bprime_source.json,robocopy.log,train_log_inherited_from_Bprime.jsonl}`; dev box `C:\Users\Admin\tsc2\R7_resume\` | 2 |
| fit probe (bs 8/4/2/1) | `repo:…/raw/fit_probe/fit_bs{8,4,2,1}{.log,_config.json,_train_log.jsonl}` | 2 |
| reads (R1–R7, computed) | `repo:…/raw/reads.json` | 2 |
| tools that produced every number | `repo:…/raw/tools/{run_arm.py,inspect_and_build_cache.py,reads.py,run_r7.sh,wait_step.py}` | 2 |
| cache index used (20 ids + geometry) | `repo:…/raw/cache_index.json`; live at `C:\Users\Admin\tsc2\cache\index.json` | 2 |
| private code mirror that ran (HEAD blob `8264ded8`) | `C:\Users\Admin\tsc2\stack` — **dev box only, by design** (a copy of tracked HEAD content; nothing to bank) | 1 |
| E arm | **not produced** (STOP rule) | 0 |

Staging: every `repo:` path above is `git add`ed (explicit paths) and verified with `git ls-files --cached`
— see the report. Nothing committed, nothing pushed, no branch switched. `GOALS_AND_CLAIMS.md` untouched.

## PROPOSED REGISTER ROW (H-TSC-2) — for the Master Mind to apply to `GOALS_AND_CLAIMS.md`

Table row (Live claims & hypotheses):

`| H-TSC-2 | An EMA target path cannot inflate (or collapse) faster than its decay allows — the EMA teacher pins refav1's tactical/strategic target scale | **OPEN — UNTESTED (E-ARCH-TSC-2 VOID on the dev-box rig)** | `…/2026-09-02-refav1-ema-inflation/RESULT.md`: B′ (the live recipe, bs 1, 250 steps) did NOT reproduce the inflation (`tgt_std_tac` ×0.98, VOID branch < 3), so E was not run per the SPEC's STOP rule; A′ collapsed as required (`adapter_std` −25.7 % by step 220; participation 27 → 9; run truncated at ≈ step 226 by a harness kill, read at the last banked row); `tgt_std_op` 0.98 → 0.98 (known value holds). Needs a bs-8 rig (Thor between checkpoints / a pod) or ≥ 1,000 bs-1 steps. |`

Prose amendment to **C-REFAV1-TAC-INFLATION** (append): *"E-ARCH-TSC-2 ran on the dev-box slice
2026-09-02 21:17–23:15 Berlin: **VOID** — at bs 1 (nothing larger fits in 8 GB; bs 1 itself runs over-committed at
12.29 GB) the live recipe's `tgt_std_tac` did not inflate over 250 steps (×0.98; TSC-1's bs-8 arm B read ×14.7 on
the same recipe), so arm E is uninterpretable and was not run. A′ collapsed as required (`adapter_std` −25.7 % by step 220; participation 27 → 9; run truncated at ≈ step 226 by a harness kill, read at the last banked row). **R7 MEASURED: an OFF
checkpoint does NOT load under `--ema-targets`** — strict `load_state_dict` refuses on 20 missing `ema.*` keys
(`refa_v1_train.py:348`) — so **PI option (b) is a code change** (non-strict resume + teacher←student init +
test), not a checkpoint switch. Tier T0, models discarded."*
