# REFe launch gate: condition C5 of `P6_GO_SIGNAL.md`, and the binding rule of CLAUDE.md (PI 2026-09-26)

**Rule.** No training run starts without a PASS token bound to the exact commit, argv sha256 and data-manifest sha256s. The chain refuses to start
`train.py` without a matching token. A token is never hand-written. Every check has a deliberate-regression arm that makes the gate FAIL, and a new trainer
flag without a G-DVB entry is refused by the gate itself.

## 0. Headline: what needs the Master Mind

1. **LAND the gate and two small edits together.** All files below are staged, not committed. The token's commit is a measured fact only after landing:
   `launch_gate.py ship-manifest --commit <40-hex>` REFUSES any shipped file that is not its blob at that commit, and the pod's `mint` REFUSES a tree that is not
   the manifest. Until the package is landed, **no token can be minted and `p2r_pod.sh` refuses at its first line** (`SHIP_MANIFEST.json` missing). That is the rule working, not a defect.
2. **Two edits outside the new files, both behaviour-preserving, both needed by the gate:**
   `refe/model.py` `REFeConfig.__setattr__` (D0: an undeclared attribute raises) and `refe/train.py` `main()` split into `build_parser()` + `main()` (a pure move of
   the 47 `add_argument` calls, so G-DVB can introspect the flags). The P6 implementer edits the same two files: merge, do not overwrite.
3. **Route (P6-G) is DECIDED in this note: a REFe-native gate with the generic gate's token schema and key, NOT a landing of the banked 7,806-line copy.** Section 1.
   If the Master Mind prefers the generic gate on the tip, the work left is listed in 1.3.
4. **`refe_p6` REFUSES today** (section 5). It stops refusing only when the P6 implementer builds D1/D5/D6 and the trainer emits the P6 loss terms. Names are PROVISIONAL.
5. **Not yet seen on a pod** (section 7): G-LIVE on the real data, the carve of the smoke bank, G-SUITE under the pod's file set. The instruments are exercised on the dev box
   at rehearsal scale; the rehearsal can never mint.

## 1. The route, and why

### 1.1 Options
| | A. land `launch_gate.py` (banked, 7,806 lines) with a `refe` profile | B. REFe-native gate, same token schema and key (chosen) |
|---|---|---|
| what REFe gets from it | the token machinery only: ~100 of its lines. Its checks load `refc_v3_train_gate`, call `tanitad.train.declared_vs_built(_v6)`, and read a `stack/` + `taniteval/` tree (`TREE_ROOTS`, `:156`) | the same token, plus checks written against `refe/train.py` |
| runs on the pod | no: the pod ships `code/ refe/ splits/` only, no `stack/`, no git (`P2R_POD_RUNBOOK.md` section 2) | yes |
| G-DVB | imports `tanitad.train.declared_vs_built`; REFe is not a `tanitad.train` trainer. `SPEC_P6` section 5.3 already says "the REFe profile needs its own declared-vs-built map" | its own table (`gate_checks.DVB_KINDS`), 47 of 47 flags |
| risk | lands an unreviewed monolith whose 95 % is refc/refcv7/v7f-specific, to use 5 % of it | ~900 new lines, all tested |
| `SPEC_P6` section 5 fallback | - | "a refe-native token with the same schema and key. Never hand-written": this is it, taken as the primary route |

### 1.2 Same schema and key, proved rather than claimed
`gate_token.py` re-implements `canonical_json`, `argv_sha256`, `sign`, `signature_ok`, `key_id`, `fingerprint` and `fp_key` byte for byte. `test_launch_gate.py` loads the
banked module by path (when present) and asserts equal outputs on sample inputs (MEASURED, dev box: passes). Token fields: `schema tanitad.launch_gate/1`, `kind launch-gate-token`,
`binding {profile, commit, tree_sha256, argv_sha256, argv, trainer, gate_inputs, data, data_sha256}`, `checks`, `reasons`, `key_id`, `hmac_sha256`.

### 1.3 If the generic gate is wanted on the tip anyway
(a) add `refe_p2`/`refe_p6` to its `PROFILES` (its `verify_token` falls back to the refc profile for an unknown name and would recompute the wrong data manifest);
(b) point `TREE_ROOTS` at `refe code splits` for those profiles; (c) route G-DVB/G-LIVE to `gate_checks.py`. Nothing in the REFe token needs to change.

### 1.4 Reuse, not duplication
`p2_gate.py` already does G1 (warm start == the planner's load), G2 (the analytic pair: undetached -> non-zero LoRA gradient; detached -> EXACTLY 0.0; `traj_head` 0.0 either way),
G3 (`--pdm-only`, with a non-inert mutation) and G4 (the real trainer, 2 steps) correctly. It is untouched. Its `gate.json` is **consumed** as check `G-P2GATE`
(`launch_gate.consume_p2_gate`: parsed, all four present and `ok`, never grepped), and its sha256 is bound into the token as `gate:p2_gate_json`.

## 2. How the checks observe the thing, not the declaration

The REAL `train.run(a)` runs **in-process** under `gate_checks.instrumented_run`. It is not reimplemented: the harness subclasses the constructors the trainer calls
(`REFe`, `AdamW`, `CosineAnnealingLR`, the three banks, the samplers, `DataLoader`) and records their arguments and results, registers a global optimiser pre-step hook that reads every
parameter group's gradient before each step, and captures the first real batch. Two runs: **BUILD** (the launch levers, stopped at the first `opt.zero_grad`: the schedule and every constructor argument are observed) and
**SMOKE** (`--steps 2 --accum 2`, the carved gate bank, workers 0: losses and gradients are observed).

| check | what must hold | regression arm (`TANITAD_GATE_ARM`) | red-arm result (MEASURED, dev box, `test_launch_gate.py`) |
|---|---|---|---|
| **G-HYG** | `REFeConfig` refuses an undeclared attribute; no non-field attribute on the built config; the trainer reads no `a.<attr>` the parser does not declare (AST of `run`/`run_identity`: 47 of 47 declared) | `no_cfg_guard` (remove the D0 guard), `undeclared_namespace_read` | FAIL "ACCEPTED an undeclared attribute"; FAIL "reads ... `undeclared_lever_g_hyg`" |
| **G-DVB** | every argv lever equals the BUILT value: 33 observed rows (backbone, trunk dir, every bank argument, init-from, `detach_scorer_context`, lr, wd, cosine and its T_max/eta_min, fused, sampler class, batch, workers, steps re-derived independently as `epochs x ceil(epoch_len/(batch x accum))`, seed, tf32, device, compile, amp dtype). `DVB_KINDS` covers all 47 parser flags: 33 built, 1 consumed (`score_w`, traced statically and through the live `score` term), 5 run-management, 8 controls/ablations (forbidden in a launch by the profile). An unknown flag FAILS; a stale entry FAILS; a flag the parser rejects FAILS ("declares a lever the trainer does not build") | `lever_ignored` (the model re-sets `detach_scorer_context` True), `unknown_flag`, `stale_registry`; also a real extra parser flag, no arm | FAIL "`--scorer-sees-trunk`: declared False != built True"; FAIL "NO declared-vs-built entry"; FAIL "no longer has"; FAIL `brand_new_lever` |
| **G-LIVE** | every declared loss term (`traj_L1`, `score`) present and finite on every logged row; each of the 12 declared trainable groups has a finite NON-ZERO gradient on some step; the 2 declared-frozen groups read EXACTLY 0 (no tensor with a gradient, none requires grad); a group in neither list FAILS; on-policy coverage > 0; and the probe's own control: on the first real batch a zero loss leaves every gradient **exactly 0.0** and a real loss does not | `zero_score` (BCE returns zeros), `frozen_group` (`traj_head` frozen after build) | FAIL "`score_dec`: gradient is EXACTLY zero ... (4,213,760 parameters never trained)"; FAIL "`traj_head` is FROZEN (0/4 tensors require grad)" |
| **G-EVAL** | the planner's load sequence (`REFeConfig.for_backbone` -> `config_for_checkpoint` -> `REFe` -> `load_for_inference`, text-checked to still be in `planner.py`) on `model_final.pt`: config fields equal (allow-list: `detach_scorer_context`, which changes gradients and no forward value), every tensor bit-identical to the trained model, forward outputs on the first REAL batch bit-identical to the trainer's own load | `planner_cfg_differs`, `loader_drops_tensor` | FAIL "planner builds a different config ... undistort"; FAIL "1 tensor(s) differ" |
| **G-CKPT** | halt after step 2, `--resume` to 4, against 4 uninterrupted: model, optimiser state, data position and the logged loss trajectory identical | `resume_drops_opt` (zeroed moments) | FAIL "optimiser state after the resumed run != the uninterrupted run's" |
| **G-SUITE** | `test_p2r_chain.py` and `test_launch_gate.py --static` exit 0 and print their token; **a SKIP FAILS** (a skip is not a pass) | `suite_failing_test` | FAIL "ZZTEST_PASS NOT seen" |
| **G-CLOCK** | P6 only: realised epochs, steps and epoch length equal the registered schedule (25, 10,063, 103,037). **NA for P2** with the reason recorded: its labels are a relabel of a banked bank, not a clock | `clock_wrong_epochs` | FAIL "realised epochs 1 != registered 2" (rehearsal-scale schedule) |
| **G-P2GATE** | `p2_gate.py` G1-G4 all present and ok (P2 only) | a `--skip-trainer` or failing `gate.json` | unit-tested both ways |

Each regression arm makes **its own** check FAIL, writes evidence tagged with the arm, and `finalize` rejects any evidence produced under an arm and refuses to mint at all while `TANITAD_GATE_ARM` is set.

## 3. The token, the chain, the refusal

* **`SHIP_MANIFEST.json`** (dev box): `python refe/launch_gate.py ship-manifest --pkg <package root> --commit <40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json`. Ship it with `code/ refe/ splits/`.
* **Stage `launch_gate`** in `p2r_pod.sh`, after `gate` (p2_gate), before the preflight: `launch_gate.py run --profile refe_p2 --args "$ARGS" ...` on the REAL argv and data. `ARGS` is now defined once, before this stage
  (the token binds this string; the previous definition site is gone; `ARGS` itself is byte-identical to `p2_chain.sh`, pinned by the existing test).
  The smoke bank is carved from the real `train_grow` rows of the gate's on-policy sets (`gate_fixture.carve_gate_bank`), so the scorer loss is reachable on every micro-batch.
* **`token_ok`** runs `launch_gate.py verify --profile refe_p2 --token ... --args "$ARGS" --pkg "$PKG"`: it RE-MEASURES the argv, the code tree (including the gate's own code) and the data fingerprints, and checks the HMAC.
  It is called after the stage, before the stage is marked done, before the preflight, and **before every `train.py` launch** (including the supervised relaunches). It is never a grep of a log.
* **Key:** `TANITAD_GATE_KEY_FILE=$P2/launch_gate.key`, created at the first run on the pod, outside the shipped tree and outside anything `p2r_sync` pushes. A token minted anywhere else does not verify there.
* **Data manifest:** a file is its sha256, a directory is its listing (relpath + size of every file, contents NOT hashed: a 449 GB bank is not re-read per launch). Same method as the generic gate.

### Chain tests (`test_p2r_chain.py`: 89 -> 111 checks, 17 -> 32 mutations; all MEASURED, dev box)
Deleting the token check before `train.py`, before the preflight, deleting the stage, making `token_ok` constant / a grep / verify the wrong argv or no profile, defining `ARGS` after the gate or twice,
grepping the gate log, not requiring the manifest, putting the key in the shipped tree, leaving fd 200 open on the gate child, never marking the stage, running the gate after training: **each goes RED**.
Behaviour: `token_ok` extracted verbatim and run under bash against a token minted by the gate's own `mint()`: matching argv -> rc 0; a changed `--lr`, an extra flag, another host's key, an edited token, a FAIL token, no token -> rc != 0.

## 4. Profiles

* **`refe_p2`**: the chain's design asserted, not trusted: `--scorer-mode onpolicy --pdm-only --scorer-sees-trunk --epochs 3 --epoch-unit scenes --batch 4 --accum 64 --lr 5e-5 --amp bf16`; forbids the controls and ablations (`--synthetic --overfit* --spread --break-optimizer --halt-after-steps --declare-change --preflight --cpu --n-cameras --no-undistort --no-cosine --weights`). **Clean on the chain's real `ARGS`** (tested).
* **`refe_p6`** (SPEC_P6 section 5): `--scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller online --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1`; forbids `--grow --grow-scenes --onpolicy-targets --init-from` as well; requires the loss terms `traj_L1 score pos_L1 yaw_L1 bce_nc bce_dac bce_ep bce_ttc bce_c bce_ddc` and the registered schedule. **Flag and term names are PROVISIONAL** until the implementer lands them (the same stance the banked gate took for NEW-2).
* **`refe_rehearsal`**: P2's levers on a 6-tuple bank carved from `data/refe_consistent`, ViT-S, one camera, synthetic images, CPU. Exercises every instrument; its evidence is tagged REHEARSAL and can never mint.
* **`refe_rls1`** (RL-S1: `PREREG_RL_S1.md`, `HANDOFF_RL_S1.md`; trained by **`scorer_finetune.py`, not `train.py`**): profiles now carry `parser_module` / `trainer_src` (every other profile: `train` / `train.py`, unchanged) and
  `runner: scorer` here. There is no BUILD/SMOKE capture (that is `train.py`'s), so the checks are `G-HYG` (the scorer's namespace reads are a subset of its parser; the D0 config guard), `G-DVB` (`DVB_KINDS_SCORER`: every one of the
  parser's 46 dests classified `data` / `built` / `consumed` / `run` / `control`, a table SEPARATE from `DVB_KINDS` which S5 pins equal to `train.py`'s 55; unknown flag FAILS, stale entry FAILS, a dead lever FAILS, an input path the token
  would not fingerprint FAILS; the observed side compares the argv's effective levers with the trainer's own `data` event), `G-PREFLIGHT` (the trainer's own `--preflight` log, PARSED: every gate the argv obliges of G1-G5, G7, G9, G10 (`--lane-head`),
  G11 (`--epo`) present and `ok is True`, a gate the argv does not oblige = another configuration, G11 re-judged from its own fields; **there is no G6**; the file's sha256 is bound into the token as `gate:preflight_json`), `G-SUITE`
  (`test_rl_reward.py`, `test_rl_epo.py`, this gate's static tests), and `G-CLOCK` NA (as P2). The ARM is a class the argv must satisfy (`rls1_refusals`): **C** (C0/C1) = no `--epo` and no EPO lever; **R** (R0/R1) = `--epo
  --epo-terms-train`, no K/L; **R2** = R + `--epo-k --epo-l`; all of them `--pdm-labels-train/-heldout --require-pdm --b-mode listnet --lam 0`, the PREREG recipe (`RLS1_RECIPE`) and the registered EPO lever values (`RLS1_EPO_REGISTERED`, the
  effective value, named or default). `run` mints for ONE named arm (`--arm-name C0|C1|R0|R1|R2`, which also asserts the arm's seed); an undeclared arm is read off the argv, which cannot notice an omitted flag. `--gate-preflight-json` takes
  `preflight.jsonl` (the `data` event is what binds the preflight to the argv's levers); `PREFLIGHT_PASS.json` alone passes the gates but G-DVB refuses it.

## 5. `refe_p6` refuses today (a test, not a note)

`launch_gate.py refusals --profile refe_p6 --args "<the P6 argv>"` exits 5 with, among others:
`DECLARED, NOT BUILT: refe_p6 declares --yaw-wrap but the trainer's parser has no such flag`, the same for `--nav-mode` and `--labeller`, and
`DECLARED, NOT BUILT: refe_p6 requires the loss term 'yaw_L1'; train.py never emits it` (and the seven other P6-only terms: `pos_L1`, `bce_nc`, `bce_dac`, `bce_ep`, `bce_ttc`, `bce_c`, `bce_ddc`).
End to end (`test_launch_gate.py`, S4): `run_checks` marks all 7 P6 checks FAIL with "profile refusal", `finalize` writes `FAIL_*.json` and **no `PASS_*.json`**, and `verify` of that token REFUSES ("token verdict is 'FAIL', not PASS").
A config that declares a lever the model does not build is a launch blocker, never a footnote (CLAUDE.md).

**What unblocks P6** (for the implementer; each item is a flag or term the profile names): D5 `--yaw-wrap {on,off}` named in argv; D6 `--nav-mode`; D1 `--labeller online`; the per-term metrics for position, yaw and the six BCE components in `metrics.jsonl`; a G-DVB row for each new flag
(`DVB_KINDS` is the table; an entry-less flag is refused by the gate itself); the schedule registered in the profile (25 epochs, 10,063 steps, 103,037) checked by G-CLOCK once `--grow` is gone. G-LIVE-LAB (labeller identity, throughput >= 1.5 x 4.49 sets/s, stall <= 5 %, lag = 0), G-LIVE-MEM (peak <= 44 GB) and G-EGR from SPEC_P6 section 5
are **not implemented here**: they test components that do not exist yet. They are added to `refe_p6.checks` when D1 and `hf_egress` integration land.

## 6. Results (MEASURED on the dev box, 2026-10-11; Python 3.13, torch 2.11, CPU)
| suite | result |
|---|---|
| `refe/test_p2r_chain.py` | `ZZTEST_PASS 111` (was 89), 0 skips |
| `refe/test_launch_gate.py --static` | `ZZLG_STATIC_PASS 90` (76 with `--no-banked`, the pod's mode) |
| `refe/test_launch_gate.py` (full, 10 m 41 s) | `ZZLG_FULL_PASS 121`: the static 90, then on the real trainer (in-process, ViT-S rehearsal bank): every check PASS clean; **7 regression arms each make their own check FAIL**; G-CLOCK PASS and 2 FAILs; the smoke-bank carve; an end-to-end `run_checks` -> `finalize` -> **a real PASS token** -> `verify` MATCH, then REFUSED on a changed `--lr`, on an edited gate source file, and on a changed PDM-label directory |
Rehearsal timing (ViT-S, 1 camera, batch 1, CPU): BUILD ~12 s, SMOKE ~15 s, G-CKPT ~60 s, the full gate on the rehearsal ~3 min.
`stack/` was not touched and no `stack` test references the refe package (searched `stack/tests`), so `stack` `pytest -q` was **not re-run**; the CLI entry `python train.py ...` was exercised after the `build_parser` split.

## 7. Pod-only, UNVERIFIED, and honest limits
* **Never run on a pod:** the real-data smoke (ViT-L, `--compile`, `--amp bf16`, `--tf32`, `--workers 2` forced to 0 in the smoke), the carve of the real `train_grow` rows (streams GBs of jsonl once), the fingerprint of the ~400 k-file pixel directory on every `verify` (a few launches: UNVERIFIED cost, ESTIMATED tens of seconds each), and G-SUITE's two scripts under the pod's file set. If `test_p2r_chain.py` prints a `ZZTEST_SKIP` there (e.g. `served_count` cannot import), G-SUITE FAILS by design.
* **Estimated pod cost of the gate:** BUILD loads the real `--targets` bank once (the preflight already does), SMOKE 2 optimiser steps, G-CKPT 3 short trainer launches, G-EVAL two ViT-L builds. ESTIMATED 15-25 min, UNVERIFIED.
* **G-DVB "consumed" is weaker than "built":** `--score-w` is traced statically and through the live `score` term, not by a value read from a built object. The `amp` row needs CUDA (on CPU the autocast is off by construction, so the declared value is derived from the device). `compile` is observed only on CUDA.
* **G-EVAL emulates the planner's loader and text-checks `planner.py` for it; it does not import `Planner`** (that pulls the NAVSIM stack). A change to the planner's constructor beyond those four lines is caught only if one of them changes.
* **The smoke argv deviates from the launch argv in a documented, minimal way:** `--epochs`/`--grow-scenes` removed, `--steps 2`, `--accum` capped at 2, `--workers 0`, the bank swapped for the carved gate bank, the on-policy side for the gate sets. BUILD keeps every launch lever (except the gate sets) so the schedule and constructor arguments are observed on the real configuration.
* **The key is pod-local.** A token minted on the pod verifies on that pod. After a pod replacement the chain re-runs the gate (delete `stage/launch_gate.ok`; a verifying token is required regardless of the marker).
* **Commit binding is as strong as the landing.** `ship-manifest --commit` proves the files equal their blobs; the pod proves its tree equals the manifest. It cannot prove the commit is on `origin`.

## 8. Run
```
# dev box, once the package is landed
python refe/test_launch_gate.py            # full: ~9 min, CPU
python refe/test_launch_gate.py --static   # ~10 s
python refe/test_p2r_chain.py              # ZZTEST_PASS 111
python refe/launch_gate.py refusals --profile refe_p6 --args "<P6 argv>"     # exits 5
# regression arm by hand
TANITAD_GATE_ARM=zero_score python refe/launch_gate.py run --profile refe_p2 ... --only G-LIVE     # G-LIVE FAIL, no token
```
