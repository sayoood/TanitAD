# refcv8 launch gate, 2026-10-14 — the G-SUITE precheck found 3 failures; all 3 FIXED, none admitted

**Context.** Launch commit `c95768b` (refcv8-r101-s0). The Thor launch-gate suite ran TIP `43c0163` against CAND
`c95768b`: TIP and CAND both read **10,042 pass / 150 fail / 267 skip of 10,459** (`raw/thor_suite_c95768b47b21_verdict.json`,
sha256 `d2c2893a…`; bound as `raw/thor_suite_c95768b47b21_suite_bound.json`). Its own verdict was PASS (nothing new on
CAND), but the GATE's judge (`launch_gate.judge_suite_verdict`, run as a precheck before the gate) admits a Thor failure
only if its id is in the pinned environment list (165 ids). **3 failures were outside it** — all pre-existing on TIP:

| test | what failed | class | disposition |
|---|---|---|---|
| `test_text_encoding_is_explicit::test_subprocess_text_mode_always_names_its_encoding` | 3 text-mode `subprocess.run` calls without `encoding=` (`stack/ops/gate/thor_gate_runner2.py` ×2, `stack/tests/test_launch_gate_refcv8.py` ×1) | REAL defect (a static check; fails on every host) | **FIXED**: `encoding="utf-8"` |
| `test_refcv8_emit_schedule::test_I2_emission_off_at_step0_…` | `max_base_score_abs_diff` 1.9e-6, asserted == 0.0 (the I2 row itself PASSES) | REAL defect, visible only on aarch64 CPU | **FIXED** at the root (below) |
| `test_refcv8_warm_start::test_allocation_and_prior_free_group_…` | `torch.equal(sel_score)` False; traj / sel_idx / base fan / u0_hat all equal | same defect | **FIXED** at the root |

## The defect (MEASURED, `raw/ulp_diag.txt`, scripts in `code/`)

`AnchoredDiffusionDecoder._r8_split` decodes the base candidates and the appended extras in separate passes so that
the base pass runs refcv7's exact GEMM shapes (`RESULT_IW.md`). It passed `x[:, sl]` — a **strided view** of the
`[B, N+M+N, S, 2]` fan. `_decode` reshapes it to `[B, N, S*2]` without a copy, so `traj_proj` received the base slice
with **batch stride 768 instead of refcv7's 320** (same values, same shape). On the Thor CPU that layout takes a
different GEMM path: the first op of the base pass (`traj_proj`) already differs by 3.8e-6, and the base confidence
by 1.4e-6. Every selection-graft term is bit-equal, and the refcv8 rank term is exactly 0 on the base slice — the
ulp is entirely the layout. On x86 (MKL) the two paths happen to agree, which is why every dev-box suite was green.

**Fix:** `fn(kv, cond, x[:, sl].contiguous(), *rest)`. Thor CPU with the fix: both files 22/22 (the two failures
pass; the control without the fix fails the same two). Dev box: 25/25 targeted, 53/53 `test_launch_gate_refcv8`.

**Guard (new, `tests/test_refcv8_warm_start.py`):** the numeric identity tests can only go RED on a layout-sensitive
host, so the precondition is now asserted where EVERY host can see it — shape, stride and contiguity of each
base-pass `traj_proj` input equal refcv7's, call for call — with a deliberate-regression twin that re-compiles the
current `_r8_split` without the `.contiguous()` (the real historical defect) and must go RED. Both pass on x86 and
on Thor.

## What it did NOT affect (scope, stated)

* **The launch path on the Thor GPU:** SPEC_WPB's I-W identity gate PASSED on the real model, Thor GPU, 2026-10-04
  (`RESULT_IW.md`, `raw/I_W.json`), and on Thor CPU with real shapes all differences were 0.0. The ulp appeared only
  at the tiny rig's shapes (d = 32, 20 candidates). The fix makes the bit-identity claim hold by construction rather
  than by the GEMM library's choice at a given shape.
* **The plan:** trajectories, selected indices and the base fan were bit-identical even before the fix.

## Why a fix and not a pin

The pinned env list exists for genuine environment failures. Adding these two ids was considered first and **was
not done**: the per-host difference had a cause in our code, and a cause is fixed, not admitted. The fix costs
re-running the map and box binding records, because `refc.py` is in both harnesses' code closure.


## 2026-10-15 — the gate PASSED on `fb47a583763f`, and refcv8-r101-s0 is training

**The first run on the launch commit (`bc2816862b8c`): 8/10 PASS, 2 FAIL** (`raw/gate_bc2816862b8c/FAIL_bc2816862b8c.json`).
Every reason is quoted from the evidence:

* G-LIVE: "101 training loss calls observed, expected 100". The smoke takes one gradient-share reading at step 50.
  Under the bf16 trunk that reading runs on an fp32 REPLAY of its step (`grad_share.fp32_replay`), which is one more
  training-mode loss call. The run records it: the step-50 metrics row carries `gs_fp32_replay: 1.0`.
* G-LIVE: "1 declared-trainable leaf module(s) received ZERO gradient: ['core.decoder.r8_rc_to_cond']". This is the
  route-checkpoint projection. `refc.py` applies it only when `r8['rc'] is not None`, and the REQUIRED `--r8-no-rc`
  (PI Q-A) feeds none.
* G-EVAL: "the run record ... was written for ANOTHER argv". The gate compared its own smoke's `config.json` with the
  RAW launch argv. The smoke re-expresses `--grad-share-every`, `--grad-share-at`, `--b1-guard-step` and
  `--r8-alloc-emit-start` inside its 100 steps. Those four were the entire difference; every model flag was equal.

**Fix (`fb47a58`, `stack/scripts/launch_gate.py`)**, gate rules only, no model code:

* The replay is counted from the run's own record. An extra call without a replay row still FAILS.
* `core.decoder.r8_rc_to_cond` is admitted in refcv8's `live_dead_admitted`, and only while `--r8-no-rc` is in the
  argv. That is the flag-gated mechanism refcv7's strategic modules use. refcv7's own table is unchanged.
* `stamps_written_for()` compares a smoke record with `smoke_argv(launch argv)` at the smoke length, and is exact on
  every model flag. ⚠ The length comes from `smoke_step_count`, exactly as the smoke derives it, and is **never read
  from the record**. The newest smoke `config.json` is G-CKPT's RESUME run (`--steps 102`). The first draft read 102,
  and re-judging the real Thor record refused it again. That is how the trap was caught before landing.
* Tests in `stack/tests/test_launch_gate_refcv8.py`. Mutation: reverting the replay count, dropping the admission,
  ignoring the smoke rule, or reading the length from the record each turns exactly one test RED.

**PASS (`raw/gate_fb47a583763f/`)**: token `PASS_fb47a583763f.json` (md5 `07d9a350d1794809825ef504f484c9bd`),
10/10 checks; per-check evidence under `evidence/`. Supporting files: `final_gate.log` (the chain: suite-bind,
G-SUITE precheck, the binding records carried, the kit `gate`), `final_gate2.sh` (the chain script),
`bind_PROVENANCE.txt` (md5s of the carried binding records), `verify.json` (MATCH), and `fire.out`. Measured in
the smoke:

* G-LIVE: 101 calls = 100 steps + 1 replay at step 50; `r8_rc_to_cond` was the only admitted dead group.
* Cost: 8.94 s/step mean (median 8.90); peak `cuda_max_mem` 31.26 GB at batch 16.
* G-EVAL: `written_for_this_argv` True under `smoke_argv`, smoke length 100.
* G-DVB: 100,598,182 parameters.
* Dev box G-SUITE-PINNED: 0 failures, 0 unregistered skips.

**Launched** 2026-10-15 02:54:15Z (04:54 Berlin): supervisor launch #1 of 4, run dir
`/home/nvidia/refcv8_run/runs/refcv8-r101-s0`. The running trainer's argv, read from `/proc/<pid>/cmdline`, equals the
gated argv token for token (sha `188aa439…`). Registry: `Project Steering/MODEL_REGISTRY.md` §16.
