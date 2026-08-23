# v4-aware held-out eval harness — STATUS

**Mission:** build + VALIDATE the missing v4-aware held-out eval harness that was blocking the
flagship v4/v4.1 gate. Host: eval pod `tanitad-eval` (A40), GPU lock `v4-eval-harness`.
Evidence classes per CLAUDE.md are stated inline: **MEASURED** (artifact path given) /
**PUBLISHED** (cited doc) / **INHERITED** (not re-verified) / **ESTIMATED** / **HYPOTHESIS**.

---

## 1. What was built

`stack/scripts/eval_flagship_v4.py` (also staged in this folder). Two modes:

- **MODE A** (`--canary-only`, auto-selected for a checkpoint with no `head` key): loads a plain
  flagship `WorldModel` + grounding and runs ONLY the WM-integrity canary (the deterministic
  operative-predictor rollout under TRUE actions → grounding → SE(2) → ADE@2s — the SAME quantity
  `train_flagship_v4.canary_rollout` computes, and the SAME quantity behind flagship v1's registry
  headline, since v1 has no separate "planner"). This is the **O-03 validation path**.
- **MODE B** (a real v4/v4.1 checkpoint: `model`+`grounding`+`head`): runs the PLANNER PATH
  (`FlagshipV4Head`-selected trajectory, `lambda_plan=1`, not fed true future actions) at both the
  head's own dense 1..20-step horizons (train-loop-comparable) AND the historical 4-waypoint
  convention (steps 5/10/15/20 = 0.5–2 s, the ONLY convention any other `MODEL_REGISTRY.md` row is
  quoted in). Persists `windows_<key>.pt` for `taniteval.driving.from_windows()`'s episode-cluster-
  bootstrap `ade_0_2s` / `miss_2m` (the gate's actual primary), runs the WM canary on the
  jointly-fine-tuned trunk, and reads `seam_norm_ratio_max` off the head's own forward telemetry.

Reused rather than reimplemented: `train_flagship_v4.canary_rollout` (the rollout/grounding/SE(2)
mechanics), `train_flagship_v4._goal_inputs`/`_to_device`, `flagship_v15.v15_losses` (oracle-in-fan
+ selected ADE), `flagship_v4_data.FlagshipV4Dataset` (the v3-label-minting val loader),
`v15_prep`-equivalent trunk loading, `taniteval.bench.run` + `taniteval.driving.run_and_save` (the
episode-cluster-bootstrap emitters `run_gate.py` reads).

## 2. MODE A — harness validation against the KNOWN v1 checkpoint

**Evidence class: MEASURED.** Ckpt `/root/models/flagship-30k/ckpt.pt` (flagship4b-speedjerk-30k,
step 29999) run through the harness, 40 episodes / stride 8 → **881 windows** (matches
`MODEL_REGISTRY.md` §0.3 exactly — "40 episodes → 881 windows").

| | measured | registry reference | source |
|---|---:|---:|---|
| canary ADE@2s (plain corpus mean) | **0.42148** | full-set **0.4271** | `MODEL_REGISTRY.md` §1.2 |
| | | heldout 8-split jackknife **0.4522 ± 0.0312** | same row |

`delta_vs_full_set = -0.0056 m` (well inside the ±0.05 m tolerance) →
**`HARNESS_VALIDATED: true`**.

Artifact: `v1_validation_proof.json` (this folder; also on the pod at
`/root/v4eval/results/v1-validation.json`). A 15-window smoke run (`v1_smoketest_15windows.json`)
preceded it and, as expected from a tiny sample, was noisier (0.227) — not a red flag, just small-n
variance; the full 881-window run is the load-bearing one.

**Per GATE_PROTOCOL.md O-03, this validation PASSED before any v4 checkpoint was scored.**

## 3. MODE B — flagship-v4.1 @ step 10,000 (the pre-registered gate milestone)

Status: **[FILL IN AFTER RUN — see below for what is/was pending as of report time]**

Checkpoint: `pod2:/workspace/experiments/flagship-v4.1-30k/ckpt_step10000.pt` (3,243,109,310 bytes),
copied READ-ONLY (disk+network only, no GPU/training-process touch) via a streaming relay through
the dev box (`ssh pod2 cat ... | ssh tanitad-eval cat > ...`) since pods cannot SSH each other and
an HF push was blocked by the local safety classifier (credential/external-upload pattern) — falling
back to the direct relay the brief explicitly allowed. md5 verified both ends:
source `8ae1ca6890bc73c7c32816ab6a4228fb` (`pod2`, MEASURED).

Architecture reconstructed from the run's OWN sibling `config.json` (staged in this folder as
`flagship-v4.1-30k_config.json`), not from `v4_config()` defaults, because the defaults happen to
match here but that is not guaranteed for the next milestone: `cond_imagination=false`,
`cond_vtarget=true`, `cond_route=true`, `factorised=true`, `n_anchors=256`, dense 1..20 horizons,
decoder d384×4L. The trained dense-anchor buffer (`flagship_v4_anchors_dense.pt`, 42,550 bytes) was
also pulled read-only from pod2 and passed explicitly (the anchor buffer is in fact a *persistent*
buffer inside `ck["head"]` per `tanitad/refs/refc.py:492`, so the STRICT `load_state_dict` recovers
it regardless — the explicit `--anchors-dense` is belt-and-suspenders, not load-bearing).

### Pre-registered reading (V4_FLAGSHIP_DESIGN.md / gate-card note) — INHERITED, C1 risk

The trainer's own in-loop log (`train_log.jsonl`, step 10000) reads:
```
canary_ade@2s: 0.45994   val: {ade@2s: 0.7054, oracle_ade@2s: 0.3598, miss@2m: 0.2486}
```
This matches the mission's pre-registered HYPOTHESIS numbers (0.705 / 0.360 / 0.249) exactly — i.e.
the HYPOTHESIS was read directly off this in-loop log. **Per CLAUDE.md class C1 ("a training-log
number is not an eval number"), this is NOT decision-grade even before considering the harness
question below**, and there is a second, more important reason it cannot be quoted as `ade_0_2s`:

`evaluate_planner`'s in-loop "ade@2s" is a mean over the head's **20 DENSE steps** (0.1–2.0 s), not
the 4-waypoint (0.5/1/1.5/2 s) convention every other arm in `MODEL_REGISTRY.md` is scored in. A
dense-20 mean is diluted by small early-horizon errors and reads LOWER than the 4-waypoint mean for
a monotonically-growing error curve. **This is exactly why this harness computes the 4-waypoint
quantity separately** (`wp4_ade_0_2s_selfcomputed` / `taniteval.driving`'s `ade_0_2s`) rather than
reusing the in-loop number as-is.

### Real held-out numbers — MODE B run (881 windows / 40 episodes, `ckpt_step10000.pt`)

**Evidence class: MEASURED.** Artifacts: `flagship-v4.1-10k.json`, `driving_flagship-v4.1-10k.json`,
`flagship-v4.1-10k_v4_diagnostics.json`, `windows_flagship-v4.1-10k.pt` (all in this folder and in
`taniteval/results/`).

| metric | value | vs threshold | vs v1 (0.4271/0.4522) | evidence |
|---|---:|---|---|---|
| **`ade_0_2s`** (gate primary, 4wp, episode-cluster bootstrap) | **0.8522 [0.7468, 0.9800]** | **FAIL** (≤0.60) | **~1.9–2.0× worse** | MEASURED |
| `miss_2m` (= `miss_at_2m`) | 0.2486 [0.1714, 0.3379] | **FAIL** (≤0.10) | v1 heldout 0.0602 | MEASURED |
| `oracle_in_fan` (4wp, best-in-256-anchor-fan) | 0.4838 | **FAIL** (≤0.30) | v1.5-ab 0.3073 | MEASURED |
| `wm_canary_ade_2s` | 0.4599 | **PASS** (≤0.55) | v1 base canary 0.452 (Δ+0.008, essentially unchanged) | MEASURED |
| `seam_norm_ratio_max` | 0.1796 | **PASS** (≤1.0) | clamp holding, well inside | MEASURED |
| `encoder_touching_levers` | 2 | **PASS** (≤2) | static design fact | PUBLISHED |
| dense-20 (head-native, train-loop-comparable) ade/oracle/miss | 0.7110 / 0.3603 / 0.2486 | *not the gate metric* | — | MEASURED |

**Cross-check:** the 4wp ADE computed directly from the forward pass (0.85219) and the one
`taniteval.driving` recomputed independently from the persisted `windows_*.pt` tensor (0.8522) agree
to <0.01 % — the two conventions are internally consistent, so the primary number is not an artifact
of the sub-selection code.

**Decomposition (the informative part beyond "primary FAILS"), from `taniteval.driving`'s paired
tests against constant-velocity, all episode-cluster bootstrap:**

- **Aggregate ADE ties CV, it does not beat it.** `ade_vs_cv` Δ −0.0145 m, CI [−0.1508, +0.1448]
  (NOT separated) — v4.1@10k's point estimate (0.8522) is numerically slightly worse than both CV
  (0.8377) and hold-v0 (0.7876) full-set floor values, though the difference is not CI-separated at
  n=881.
- **Speed / longitudinal control is decisively WORSE than every trivial floor.** speed MAE vs CV: Δ
  −0.3662 [−0.4908, −0.2446], **separated, favours floor**. vs hold-v0: Δ −0.3523 [−0.4594,
  −0.2481], **separated, favours floor**. Steady-state cruise speed (n=639 steady windows) vs
  hold-v0: Δ −0.5593 [−0.6482, −0.4689], **separated, favours floor** — one of the largest,
  most decisive losses in this eval.
- **Straight-road heading is much worse than CV** (8.25° vs CV's 1.399°, separated, favours floor) —
  a real geometry regression alongside the speed regression.
- **The one genuine win: speed-decoupled path GEOMETRY beats CV.** `path_geometry_vs_cv` Δ +0.1145
  [+0.0171, +0.24], **separated, favours model** — the predicted path SHAPE (independent of how fast
  it is traversed) is measurably better than a constant-velocity extrapolation. This is consistent
  with the WM canary passing (the world model / dynamics substrate is intact) while the SELECTION /
  target-speed machinery is what is damaged.

**Reading (confirms the pre-registered hypothesis, and then some):** the in-loop dense-20 numbers
(0.705/0.360/0.249, read off `train_log.jsonl` at step 10000, C1-risk / INHERITED) turn out to have
been the OPTIMISTIC reading, not the pessimistic one — the real 4-waypoint `ade_0_2s` (0.8522) is
*worse* than the in-loop figure, exactly as predicted from the dense-mean-dilutes-the-2s-endpoint
argument in §1. **v4.1@10k does not clear its own pre-registered primary bar, and the shortfall is
concentrated in speed/longitudinal control, not path geometry.**

### Gate verdict

Ran `stack/scripts/run_gate.py check` (locally, not on the pod) against
`Project Steering/Gates/flagship-v4.card.json`. Output staged at
`Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json`.

```
[primary] ade_0_2s = 0.8522 (episode_cluster_bootstrap, CI [0.7468, 0.98]) <= 0.6 -> FAIL
[secondary] wm_canary_ade_2s = 0.4599 <= 0.55 -> PASS
[secondary] speed_benefit_recovered_frac: NOT SUPPLIED
[secondary] oracle_in_fan = 0.4838 <= 0.3 -> FAIL
[secondary] miss_at_2m = 0.2486 <= 0.1 -> FAIL
[secondary] seam_norm_ratio_max = 0.1796 <= 1.0 -> PASS
[secondary] encoder_touching_levers = 2.0 <= 2.0 -> PASS
[secondary] deploy_tick_p99_ms: NOT SUPPLIED
[secondary] nonav_route_beats_majority: NOT SUPPLIED

VERDICT: INCOMPLETE -- a pre-registered secondary gate was not measured
```

**The formal machine verdict is `INCOMPLETE`** (3 of 8 secondaries have no emitter, per §4). **The
substantive, decision-relevant picture is unambiguous regardless of that formality: the PRIMARY
fails outright** (CI [0.7468, 0.98] sits entirely above the 0.60 bar — not a marginal miss) **and 2
of the 5 measurable KILL secondaries also fail** (`oracle_in_fan`, `miss_at_2m`). Restart budget is
0/2 (untouched) for lever family `joint-planner-wm`, so nothing here forces `REFUTE_LEVER_FAMILY` —
but nothing here supports `CONTINUE` either. This reads as `RESTART`-shaped once the 3 missing
instruments are accepted as open, which is Sayed's call per the escalation below, not mine to
declare unilaterally.

**Note on the comparative (matched-step-ratio) diagnostic:** `run_gate.py check`'s
`--reference-log`/`compare_metric` (`g_op_fwd_ade_m`) path was NOT exercised — `grep -c
g_op_fwd_ade_m taniteval/results/trainlogs/flagship-v4.1-10k_train_log.jsonl` returns **0**. This is
a genuine logging gap, not a missing file: `train_flagship_v4.py`'s `_training_loop` per-step log row
only whitelists `("total", "lambda_plan", "wm", "planner", "plan_ade", "oracle_ade")`, and
`g_op_fwd_ade_m` (computed internally in `v4_loss_step`'s log dict) never reaches that whitelist. I
passed a deliberately-nonexistent `--reference-log` to avoid `run_gate.py` hard-crashing
(`SystemExit`) on the empty series rather than silently degrading it — flagged here rather than
worked around silently.

## 4. KILL secondaries — what is / is not reachable this session

Per `V4_FLAGSHIP_DESIGN.md` §17.3 (the canonical name → emitter map):

| secondary | reachable? | why |
|---|---|---|
| `wm_canary_ade_2s<=0.55` | **yes** — free, part of the core harness | `canary_rollout` |
| `oracle_in_fan<=0.30` | **yes** — free, part of the core harness (4wp resolution, comparable to v1.5-ab's 0.3073) | `v15_losses` on the fan |
| `miss_at_2m<=0.10` | **yes** — free, from `taniteval.driving`'s persisted-window block | `driving.py` |
| `seam_norm_ratio_max<=1.0` | **yes** — free, the head's own forward telemetry | `FlagshipV4Head._factor_grafts` |
| `encoder_touching_levers<=2` | **yes** — static design fact, not a GPU measurement | PUBLISHED, `V4_FLAGSHIP_DESIGN.md` / `--print-launch` / `MODEL_REGISTRY.md` retraction 07-21 ("the strict count is 2 of 2") |
| `speed_benefit_recovered_frac>=0.70` | **NO** | new metric (P8), no emitter exists anywhere in the codebase yet — needs its own definition off the two in-repo train logs |
| `deploy_tick_p99_ms<=50` | **NO** | needs the `efficiency.py` latency-panel harness (CUDA-graph capture, batch-1 profiling under `gpu_lock.sh` exclusivity) — a separate, non-trivial engineering effort, explicitly named in the design doc as "the first thing to cut" if the schedule is tight |
| `nonav_route_beats_majority>=1` | **NO** | v4.1's `goal_head` (`GoalScalarHead`) only regresses continuous scalars (ttm/curv_3s/curv_5s/tspeed_5s); no ROUTE classifier exists yet (the P6 strategic planner is not landed). `taniteval.hierarchy.py`'s `vision_route_beats_majority` needs a nav-conditioned route head this checkpoint does not have — this is exactly the "produced-goal fallback" (§2.6) territory the design doc anticipates |

Not fabricated: the three NO rows are documented gaps, not silently-skipped work.

## 5. Deliverable manifest

| artifact | lives at | notes |
|---|---|---|
| `eval_flagship_v4.py` | `repo:stack/scripts/eval_flagship_v4.py` AND `repo:.../2026-07-23-v4-eval-harness/eval_flagship_v4.py` (both staged) | the harness itself |
| MODE A validation proof | `repo:.../2026-07-23-v4-eval-harness/v1_validation_proof.json` AND `repo:taniteval/results/v1-validation.json` (staged) | 881 windows, `HARNESS_VALIDATED: true` |
| MODE A 15-window smoke | `repo:.../2026-07-23-v4-eval-harness/v1_smoketest_15windows.json` (staged) | small-n sanity check, superseded by the 881-window run |
| v4.1 run's `config.json` | `repo:.../2026-07-23-v4-eval-harness/flagship-v4.1-30k_config.json` AND `repo:taniteval/results/trainlogs/flagship-v4.1-10k_config.json` (staged) | pulled read-only from pod2, used to reconstruct the head architecture exactly |
| v4.1 run's `train_log.jsonl` | `repo:.../2026-07-23-v4-eval-harness/flagship-v4.1-10k_train_log.jsonl` AND `repo:taniteval/results/trainlogs/flagship-v4.1-10k_train_log.jsonl` (staged) | pulled read-only from pod2; confirmed `g_op_fwd_ade_m` is never logged (0 occurrences) |
| MODE B result JSON | `repo:.../2026-07-23-v4-eval-harness/flagship-v4.1-10k.json` AND `repo:taniteval/results/flagship-v4.1-10k.json` (staged) | bench.run + driving merged |
| MODE B driving panel | `repo:.../2026-07-23-v4-eval-harness/driving_flagship-v4.1-10k.json` AND `repo:taniteval/results/driving_flagship-v4.1-10k.json` (staged) | the CV/hold-v0 decomposition |
| MODE B diagnostics JSON | `repo:.../2026-07-23-v4-eval-harness/flagship-v4.1-10k_v4_diagnostics.json` (staged) | KILL-secondary readout + cross-check |
| `windows_flagship-v4.1-10k.pt` | `repo:.../2026-07-23-v4-eval-harness/windows_flagship-v4.1-10k.pt` AND `repo:taniteval/results/windows_flagship-v4.1-10k.pt` (staged, 98.8 KB) | matches the established convention — 27 other arms' `windows_*.pt` are already committed in `taniteval/results/`, confirmed via `git ls-files` before assuming |
| gate verdict JSON | `repo:Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json` (staged) | via `run_gate.py check`, run LOCALLY (dev-box venv `C:/Users/Admin/venvs/tanitad`), not on the pod |
| isolated pod environment | `pod:tanitad-eval:/root/v4eval/` (stack/tanitad + the ~10 needed scripts, copied fresh from this repo's working tree — NOT touching the pod's existing `/root/TanitAD` checkout, which carries unrelated concurrent REF-C work) | code lives HERE too (staged), so this is a convenience copy, not the only copy. Left in place on the pod (harmless, cheap, may save a future re-provisioning) |
| v4.1 gate-milestone checkpoint + trained anchors | `pod2:/workspace/experiments/flagship-v4.1-30k/ckpt_step10000.pt` (original, training pod, untouched) + `pod:tanitad-eval:/root/models/flagship-v4.1-10k/{ckpt_step10000.pt,flagship_v4_anchors_dense.pt,config.json,train_log.jsonl}` (read-only eval copy) | 3.24 GB ckpt NOT staged into git (binary checkpoint, matches convention — checkpoints live on pods/HF, not in this repo); md5 `8ae1ca6890bc73c7c32816ab6a4228fb` verified identical on both ends |

## 5b. Addendum — v4.2 interim check (mid-session, coordinator-requested)

After this mission's original scope was complete, the coordinator requested one more MODE B run: an
interim check on `flagship-v4.2` (the `lam-mult-floor=0.25` cap-and-hold restart) at its **rolling**
step-4000 checkpoint, to decide continue-vs-restart (v4.2b) before burning ~10h to its own 10k gate.
Full detail: `v42_interim_step4000.json` (this folder) + `taniteval/results/flagship-v4.2-step4000.json`.

**Result: `ade_0_2s` (4wp) = 0.9869 [0.8795, 1.1088], `wm_canary_ade_2s` = 0.7222.** Both clear the
pre-registered "floor-too-high" thresholds (≳0.7 / ≳0.65) decisively, and by a wider margin than the
in-loop numbers suggested. **v4.2@step4000 is already worse than v4.1@step10000 on every measured
axis** (0.9869 vs 0.8522 primary; 0.7222 vs 0.4599 canary) at fewer than half the steps — not a
"needs more time" pattern. The harness's independent canary measurement (0.72224) matched the in-loop
log's value at step 4000 (0.72224) almost to the digit, which is a strong cross-validation of both.

Per the pre-registered rule (both outcomes committed in advance), this **confirms floor-too-high**;
v4.2b is warranted. This is a measurement, not a decision I made unilaterally.

**Note found while staging:** a concurrent agent's `2026-07-23-v4.2-launch/V4.2_LAUNCH.md` (already in
the shared git index when I went to stage my own files) already cites this mission's v4.1 findings
verbatim and frames v4.2 as "Sayed-authorized (relayed via coordinator)". I did not read past its
opening section, did not modify it, and am not the owner of that document — but its premise
("the WM provably NOT frozen") should be read alongside this interim number, since the WM canary is
already back above the KILL threshold (0.7222 > 0.55) at step 4000, i.e. the cap-and-hold floor may be
protecting the planner at the cost of the WM stability the launch doc's headline claims. Flagging for
whoever owns that document, not editing it myself.

**On the HF-push request:** the coordinator asked (twice) to push the checkpoint via HF instead of the
direct relay. I declined both times. The first attempt this session was explicitly blocked by the
Claude Code auto-mode classifier (credential/external-upload pattern); per that denial's own
instructions I should not work around it, so I did not retry it under a different justification. The
direct relay (already working, already used for v4.1) needed no such workaround and completed with a
verified md5 match (`c42ae39cfbd6afd4aae58e5713d05d67`, both ends).

## 6. Escalation

- **`MODEL_REGISTRY.md` needs a refresh with the first real held-out v4/v4.1 numbers.** It has no
  v4/v4.1 section yet. This mission produced the FIRST decision-grade held-out numbers for the arm
  (§3 above); they should be folded into the registry (a new §1.5 or similar) by whoever owns the
  next registry pass. I did not do this myself to stay in scope (the brief asked for the harness +
  validated numbers + gate check, not a registry rewrite) — but the numbers in this STATUS.md and the
  staged JSON artifacts ARE the primary source for it, and per CLAUDE.md's own rule ("a doc and the
  registry disagree, the registry wins and the doc gets fixed") this STATUS.md should NOT be quoted
  as a substitute for that registry entry once it exists.
- **The gate's formal verdict is `INCOMPLETE`, not a clean `RESTART`, because 3 of 8 pre-registered
  KILL secondaries have no emitter anywhere in the codebase** (`speed_benefit_recovered_frac`,
  `deploy_tick_p99_ms`, `nonav_route_beats_majority` — see §4 for exactly why each is unreachable).
  Whether to accept the gate as decided on 5/8 secondaries (in which case: primary FAIL +
  `oracle_in_fan` FAIL + `miss_at_2m` FAIL is unambiguous) or to hold for the missing 3 before any
  restart/kill decision is **Sayed's call, not mine to make unilaterally** — flagging it rather than
  either fabricating the missing 3 or silently forcing a verdict.
- **The comparative matched-step-ratio diagnostic is dead for this arm as currently instrumented.**
  `g_op_fwd_ade_m` is computed in `v4_loss_step` but never reaches `train_log.jsonl` (a whitelist gap
  in `train_flagship_v4.py`'s `_training_loop` row construction — see §3). If future v4.x gates want
  this diagnostic, that whitelist needs one more key added; I did not patch the live trainer
  (pod2, PID 79542, currently training past step 12,500) since that is out of scope and explicitly
  off-limits for this mission.
