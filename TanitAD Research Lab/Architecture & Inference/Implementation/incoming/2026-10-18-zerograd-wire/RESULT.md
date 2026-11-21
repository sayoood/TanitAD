# ZG-WIRE -- the post-training zero-gradient gate is now called (2026-10-18)

Tip 821915a7fefea3fe6d68d2a493ab39ada4febd61. Closes BACKLOG `ZG-WIRE` items (1), (2) and the read-only check of (3).
Nothing here is committed or pushed: every file sits under the OUT dir at its repo path (`paths.json`).

## Integration the Master Mind must do
Land the 8 files in `paths.json` together (6 edited, 2 new tests) plus this package. They are one change: the
supervisor hook (`sup_refcv7.sh`), the trainer payload (`refc_v3_train.py`) and the gate (`zerograd_gate.py`) are
tested against each other, and two existing tests pin the checkpoint key set (`test_refc_v3_save_before_eval.py`,
`test_refcv8_ema_guard.py`) and had to move with the payload. `test_launch_gate.py` drives the real
`sup_refcv7.sh` with a fake trainer that writes no `ckpt.pt`; the rig now sets `ZEROGRAD_GATE=off` (see below).

## What the supervisor does now (`stack/ops/sup_refcv7.sh`, rule G5)
The gate runs on `${OUT_DIR}/ckpt.pt` just BEFORE the done-marker, on every completion path (loop-top
"already complete", after the trainer exits, and the new finalize-only path). The verdict is the gate's JSON, never
its exit code (a Python traceback also exits 1, so exit 1 alone cannot tell a crash from FAIL).

| gate JSON + exit code | supervisor status | done-marker | marker written | exit |
|---|---|---|---|---|
| `PASS` + 0 | PASS | written, with a `zerograd_gate` record | `summary.json` | 0 |
| `ERROR`/`no_optimizer_state` + 2 | NOT_APPLICABLE ("not applicable (no optimizer state)") | written, with that record | `summary.json` | 0 |
| `ZEROGRAD_GATE=off` | DISABLED (loud log line) | written, with that record | `summary.json` | 0 |
| `FAIL` + 1 (untrained param / failed control) | FAIL | **refused** | `ZEROGRAD_FAIL.json` (embeds the gate JSON) | 6 |
| anything else (unreadable ckpt, no JSON, JSON and exit code disagree, gate script missing, crash) | UNEVALUATED | **refused**, same as FAIL | `ZEROGRAD_FAIL.json` | 6 |

* **Fail closed.** A gate that did not run is not a pass. `ZEROGRAD_GATE=off` is the only (logged, recorded) opt-out;
  a typo in it REFUSES (exit 2) instead of disabling.
* **Allow-list** for frozen-by-design parameters: `ZEROGRAD_ALLOW` (bash array, or one string with one regex per line)
  in the manifest, `re.fullmatch` against the gate's parameter names; also `ZEROGRAD_CONTROL` (a trained parameter
  must match), `ZEROGRAD_EXCLUDE`, `ZEROGRAD_CKPT`. Every pattern is logged and recorded in the marker, with the
  gate's `unused_allow`. They are `unset` before the manifest is sourced, so a caller's environment never leaks in.
* **No relaunch on ZEROGRAD_FAIL (the resurrection trap).** A relaunch only follows an INCOMPLETE artifact; the gate
  runs only on a complete one and `finalize_run` returns to an `exit`, never into the loop. A later supervisor start
  checks `ZEROGRAD_FAIL.json` first and exits 6 without launching or re-gating; the operator must delete the file
  deliberately to re-gate. Proven by tests, including the restart variants (below).
* **Found while reading the relaunch logic:** the TRAINER writes its own `summary.json {"done": true}` at completion
  (`refc_v3_train.py`, end of `train`), before the supervisor can gate. The old startup test "done-marker present ->
  FINISHED, exit 0" would therefore have skipped the gate whenever the supervisor died between trainer completion and
  its own marker, and after an operator deleted `ZEROGRAD_FAIL.json`. A done-marker WITHOUT a `zerograd_gate` record is
  now "gate it, finalise it, never launch" (finalize-only mode).
* **Residual hole, by design (escalated, not fixed):** on FAIL the trainer's own `summary.json` is left untouched
  (it is the trainer's artifact). A downstream consumer that keys on `summary.json` alone is NOT protected; it must also
  require `ZEROGRAD_FAIL.json` to be absent. I did not find such a consumer in the repo, but I did not search pod-side scripts.
* `ckpt_step` below the requested steps (or absent) is logged as a warning and recorded (`ckpt_step_is_final`); the
  verdict stands but is then about an earlier checkpoint.

## Trainer + gate
* `refc_v3_train.py`: new `_opt_param_names(model, opt)`; `param_names` is added to both checkpoint payloads (the loop's
  inline save and `_ckpt_payload`, so the B1 re-save carries it). It walks `opt.param_groups` in order, the way
  `Optimizer.state_dict()` numbers ids -- NOT `named_parameters()` order, which differs under `--opt dd` (encoder group
  first; MEASURED on the real synthetic rig: dd registers 36 encoder + 115 head tensors).
* `zerograd_gate.py`: already preferred `ck["param_names"]` at the tip (and `test_exact_naming_from_param_names...`
  pins it), so "teach the gate to prefer it" needed no change. Added: `GateError.kind` and `error_kind` in the ERROR
  JSON (the supervisor tells "no optimizer state" from "could not look"); an uncaught crash now exits 2, not the 1 of a
  FAIL; `alignment.param_names_source`; embedded names that do not fit are REJECTED loudly and the gate falls back to
  the shape alignment (the verdict never depended on naming); an explicit `--param-names` that does not fit stays an error.
* MEASURED (synthetic 2-step `--opt dd` trainer run, `raw/gate_on_synth_trainer_ckpt.json`): ckpt keys
  `data_pos, model, opt, param_names, step`; 151 names; the gate resolves naming as `param_names`, unique; verdict FAIL on
  `core.route_head.{weight,bias}` (195 params, case a). That config has no `--no-strategic`; the production argvs
  (`stack/ops/runs.d/refcv7-*.argv.json`, `refcv8-*.argv.json`) carry `--no-strategic`, and the source says
  (`refc_v3.py` `BYPASS_DECLARATIONS`) such modules are frozen so they leave the `--opt dd` optimizer -- INHERITED from
  that comment, UNVERIFIED on a real refcv7/8 checkpoint. First real use should read the ckpt of the finished refcv7 run
  by hand before trusting a PASS or a FAIL on a production arm. An `--opt adam` arm registers frozen params
  (`model.parameters()`), which the gate sees as untrained: set `ZEROGRAD_ALLOW` for those.

## Tests (all CPU, this box: Windows, Git-for-Windows bash 5.3, venv torch 2.11)
* New `stack/tests/test_sup_refcv7_zerograd.py` (17): drives the REAL `sup_refcv7.sh` with a fake launch gate + fake
  trainer + the real gate. Clean -> DONE with a PASS record; untrained head -> exit 6, `ZEROGRAD_FAIL.json` naming
  `dead.weight`/`dead.bias`, ONE launch, no "relaunching", with and without the trainer's own summary, and again after a
  restart with the marker kept, and after the marker was deleted; trainer-written summary + fresh supervisor -> gated with
  ZERO launches; no-optimizer-state -> NOT_APPLICABLE; five "cannot look" cases -> UNEVALUATED/6; allow-list (array,
  string, partial -> still FAIL); failed control; `off` and the typo; no env leakage; stale-ckpt warning; and the RED arm.
* New `stack/tests/test_zerograd_trainer_payload.py` (5): id order vs `named_parameters()` under dd; the real trainer's
  ckpt (adam and dd) carries names that fit the optimizer state id for id and the gate resolves them as exact; embedded
  names preferred / rejected / absent; `error_kind`; crash exits 2.
* MUTATION, MEASURED (`raw/mutation_evidence.md`): hook line replaced by `ZG_STATUS=PASS` -> **17 of 17** supervisor
  tests go red; restoring the old "done-marker present -> exit 0" startup test -> the two finalize-only/restart tests go red;
  trainer names replaced by the naive `named_parameters()` list -> 3 of 5 payload tests go red; the 5 payload tests also
  fail on the tip export (no `param_names`, no `error_kind`).
* Base (tip export) vs edited tree, every test file that names an edited file or imports one that does
  (118 files at base, 120 with the two new): **2,451 passed / 9 skipped / 12 failed -> 2,473 / 9 / 12; 0 regressions**.
  The 12 failures are identical in both (`test_refcv6_diffusion` x3, `test_p10_pose_prior` x2, `test_openloop_suite` x4,
  `test_p4_p13_p14_wiring` x3: environment/data on this box, pre-existing). `raw/regression_compare.{md,json}`, per-test
  `raw/base_tip_per_test.json` and `raw/edited_per_test.json`. `taniteval/tests` (6 files naming `refc_v3_train`) 91 passed both; `tools/tests/test_superset_check.py` 13 passed both.
  NOT run: the other ~475 stack test files (they name none of the edited files); an earlier attempt to run all 594 in
  20 processes was aborted for the memory cap -- the numbers above come from the 118-file set, 4 shards per tree.
* UNVERIFIED: Linux/Thor bash (`flock` is stubbed in the tests on this box; bash 5.x idioms `${a[@]+"${a[@]}"}`,
  `<<<`, `${l%$'\r'}` are used); the 8 GB-class checkpoint load time on Thor (the gate took 2.4 s on a 1.3 GB ckpt, MEASURED
  in `2026-10-18-zerograd-gate/RESULT.md`).

## Item 4 -- `ddv2_rl_refcv5.py` (source read only, not run)
Its final `ckpt.pt` is the BASE checkpoint (`torch.load(a.ckpt)`) with the model tensors overwritten by the RL-trained
ones plus a `ddv2_rl` record (`ddv2_rl_refcv5.py` ~l.646-651); the RL run's own AdamW (l.593) is never saved. So it carries
an `opt` key only if the input checkpoint did, and that state is the SFT run's, not the RL run's: a gate PASS or FAIL on
it would be about the wrong training run. `ddv2_rl_refcv7.py` writes the RL optimizer state only in the resumable
`ckpt_latest.pt` (`opt` + `trainable`, id order == `trainable` key order, so `--param-names` could be exact); its exported
final file is weights-only (gate exit 2, as the backlog says). Neither is wired. Recommend: gate `ckpt_latest.pt` of the RL
trainer, never the exported file.

## Siblings not wired
`grep write_done_marker`: only `stack/ops/sup_refcv7.sh` is live. `sup_refcv6.sh` (and the `fix/` copy) are banked
artifacts under `Research/`. `stack/scripts/sup_refcv3.sh`, `sup_refcv5.sh`, `supervise_run.sh` write no marker of their
own (they stop on the trainer's `summary.json`/metrics), so there is no marker to refuse; they would need a different
design (a post-run hook), not this one.

## Base blob shas of the edited files (tip 821915a7)
| file | base blob |
|---|---|
| `stack/ops/sup_refcv7.sh` | 280fb0abbaea96c4c9ec6bb8cadf409a36675a3e |
| `stack/scripts/refc_v3_train.py` (CRLF file, EOL preserved) | 3bb231526026d33b0816745a21d7401df645ffed |
| `stack/scripts/zerograd_gate.py` | 90edf4c1499a1fb26d2fc9a6ad734a4d21d9df80 |
| `stack/tests/test_refc_v3_save_before_eval.py` | 793f7d92aa4dfa9472cef5a9557934c2d8af8f5a |
| `stack/tests/test_refcv8_ema_guard.py` | 56a253d5dfa52ff40d9ef7b6434777080c235c53 |
| `stack/tests/test_launch_gate.py` | 65ad0819b48b339f83b130d4af29a6ed2293dc88 |

`raw/diffs_vs_tip_821915a.patch` holds the six diffs (EOL-normalised, for review).
Suggested BACKLOG `ZG-WIRE` update after landing: (1) and (2) done, (3) checked (`ddv2_rl_refcv5.py` = stale base opt;
`ddv2_rl_refcv7.py` = gate its `ckpt_latest.pt`); open: first read on a real refcv7/8 checkpoint, consumers of `summary.json`.
