# DRAFT_RECOVERY_CARD: RECOVER-FT, a recovery fine-tune of O from DAG-OACC (ChauffeurNet-style)

⛔ **DRAFT. NOT FROZEN. NOT LAUNCHED.** Nothing here trains, queues or touches box 2. It exists so a PI go on box 2 can be answered the minute HISTSWAP is read.
Gate to freezing: (a) `HISTSWAP` reads SUPPORTED (or MIXED with its leak-free arm run) and (b) the PI's box-2 go (asked 17:3x on 2026-11-02, no answer at writing). Every unknown is marked **UNVERIFIED**. Evidence classes: MEASURED (ours, path given) / INHERITED (not re-verified) / HYPOTHESIS.

## 1. The idea (HYPOTHESIS) and the prior art that must be read first
O follows its own slowed ego history (H-HISTSWAP-1 tests this at zero training). The remedy under test: at training time, perturb the measured ego-history input along a kinematically consistent deviation, re-express the GT future from the perturbed state as a recovery target back to the logged path and speed, keep the images as logged;
inference inputs are unchanged (vision plus measured ego history); labels may use GT (BINDING: labels may use ego, inference is vision-only).

⚠ **This family has already been run in the programme (ORC, DAG, DAG2, DAG-ORC): ORC, DAG2 and DAG-ORC each read HARMS under their own tables, and DAG-ORC's crawl rows (M1 among them) did not clear.** A GPU hour spent here needs an answer to *why this one would differ*:

| arm | what it did | read | artifacts |
|---|---|---|---|
| **ORC / ORC0** (O-RECOVER) | from M2b, 3,000 steps, `--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0`; history speed scaled down, **targets untouched** (logged future in the NOW frame); ORC0 = p 0 control | **HARMS** under its own pre-registration's table (`OUTCOME.OUTCOME` in `v8f-orecover-t1/read/ORC_READ.json`, MEASURED by that reader; its rows were not re-read here) | `E:/Projects/TanitAD-artifacts/v8f-orecover/` |
| **DAG** (DAgger iteration 1) | the model's OWN T1-kin states swapped in, labels at the matched route point | ROBUST (T heads); adopted as base | `RESULT_V8F_DAGGER.md` |
| **DAG2** (iteration 2, DAG's own states) | same, one more iteration | **HARMS** | `dag2_result/RESULT_V8F_DAGGER2.md` |
| **DAG-ORC** (ORC composed into DAG-OACC's recipe, p 0.3) | `--ego-perturb-with-dagger` | **HARMS**; crawl rows cleared 1 of 4 (s0, yaw-rate only) and 0 of 4 (s1); ADE6 +0.1434 m (s0, separated-worse) | `dagorc_result/RESULT_V8F_DAG_ORC.md` |

**What this card changes** (all HYPOTHESIS): (i) the deviation law is EMPIRICAL (resampled from the model's own executed-minus-logged deviations, T1-kin ticks or DAG's bank) instead of a hand-set U(0.5, 1) speed scale: a 32-window smoke of the HISTSWAP tool (DAG-OACC s0, all speeds, `logs/smoke_histswap.out`) shows a mean |v0_in - v0| of 0.525 m/s at ticks >= 1, far below a 50 % scale cut at 3-8 m/s;
(ii) a lateral / heading offset that moves the NOW anchor, with the target re-expressed (UNBUILT: ORC shifts only the heading history and keeps NOW at the logged pose); (iii) start from DAG-OACC with a matched zero-perturbation control, so "more training" is separated from "perturbation".

## 2. Where things are (file:line from tree `dagt1_tree_b43c7a9a`, trainer sha256 `38e0a157...`, 1,676 lines; the box-2 trainer at commit `aeb54468` is 3 lines longer at the log line (1641 vs 1638), so lines are +-3: UNVERIFIED against `aeb54468`)
* **Trainer for O:** `stack/scripts/v8f_mini_train.py` (box 2: `D:\tanitad\repo\stack\scripts\v8f_mini_train.py`). `main` :1435; `--init-ckpt` :152 (strict, config-checked, fresh optimiser); `--ego-perturb-*` :125-136; `--dagger-*` :139; `FLAG_CHECKS` :67 (`init_ckpt` :93, `ego_perturb_*` :87-88);
  `make_ego_perturb` :309; `bad_ego_perturb` :315; DAG x ORC exclusivity :426; `declared_vs_built` :570 (appends `bad_ego_perturb` / `bad_dagger`); `ego_perturb_probe` :934 (reach + a p = 0 control that reads EXACTLY 0.0); `ego.apply` in the loop :1599.
* **Perturber:** `stack/tanitad/data/v8f_mini_data.py` `EGO_PERTURB_DEFAULTS` :580, `EgoPerturb` :593 (law :596-603; rebuilt tensors :584; targets never touched :605). DAgger: `stack/tanitad/data/v8f_dagger.py` `match_route` :124, `reanchor` :162-176.
* **DAG-OACC (the init), MEASURED:** argv `E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{0,1}_dagoacc/argv.json` (sha16 `0ee46fc1a9a0493b`): `--init-ckpt M2b_s`, 3,000 steps, lr 1e-4, warmup 100, batch 64, `--p-gt-start/-end 0.2`, `--ego-perturb-p 0`, `--dagger-p 0.1`, `--dagger-bank .../dagger_bank/s{s}` (sha256 `197a68bb...` at s0), `--w-acc 49.6969`.
  Gate token commit `aeb544683b1c`, checkpoints `E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{0,1}_dagoacc/ckpt_last.pt`.
* **Gate:** the v8f family's G-DVB is `FLAG_CHECKS` + `declared_vs_built` in the trainer, run by `--selfcheck-only` (the DAG-OACC token's `gate.kind`); `stack/scripts/launch_gate.py` carries a `_v8f_profile` :966. Whether that profile needs a new entry for recovery flags: **UNVERIFIED**.

## 3. How ego history enters O (MEASURED by reading `stack/tanitad/models/v8f_mini.py`)
`run_o` :698-712: O's ego conditioning is ONE row, `self._ego(self._at(b["p10f"], i))` (the 8 kinematic features at the tick frame, :637-642), and the base plan is `self.prior(self._at(b["p10f"], i), b["base"])` :710; the issued plan is `upsample_prior(P) + residual` (`head_o` x basis).
`hist` = 4 frames only feed the scene tokens; `pose_t` / `pose_s` re-express held T / S outputs. So a perturbation must rebuild `p10f`, `base`, `kin3`, `p10_t`, `pose`, `pose_t`, `pose_s`, `nav` together (`EGO_PERTURB_REBUILT` :584 already does for speed / accel / yaw rate); a lateral offset additionally moves `pose_now` (the anchor), which ORC never does.

## 4. Proposed design (DRAFT; every parameter UNVERIFIED until frozen)
| arm (per seed s = 0, 1) | argv vs DAG-OACC_s | role |
|---|---|---|
| **R_s** | `--init-ckpt` DAG-OACC_s ckpt, 3,000 steps, lr 1e-4, warmup 100, same data / `--w-acc`; `--recover-p 0.3 --recover-dev-bank <bank> --recover-dev-bank-sha256 <sha> --recover-target rejoin --recover-rejoin-steps 10 --recover-seed s`; no DAgger flag | the lever |
| **C0_s** | the same argv with `--recover-p 0` (flags named, lever off) | zero-perturbation control: same steps, same init, same optimiser: "more training" |
| **C0rep_s** (optional, PI cost call) | C0_s with `--seed 10 + s` | same-init replicate floor (the run-to-run floor, `H-ESTIM-SEED-1`) |

* **Deviation bank (HYPOTHESIS):** per (window, tick >= 1) executed-minus-logged state of the model's OWN T1-kin rollouts: speed `dv`, accel `da`, yaw rate, lateral `dy` and heading `dpsi` offset (computed from the ticks file's executed `pose` and the logged pose at `j`; the Y runs `SREP_*` already hold them). Taken from Y (self-driven), NOT from HISTSWAP's X (its executed states are the swapped regime).
* **Target (HYPOTHESIS):** the logged future re-expressed in the perturbed frame, with the first `--recover-rejoin-steps` knots blended (smoothstep) from the straight continuation of the deviated heading onto the logged points, then logged speed profile. `--recover-target logged` (pure frame change, no blend) is the matching switch.
* **New flags and their G-DVB entries (names provisional):** `FLAG_CHECKS` gets `recover_p`, `recover_dev_bank`, `recover_dev_bank_sha256`, `recover_target`, `recover_rejoin_steps`, `recover_seed`, each `"rec:<name>"` read back from the BUILT perturber's `built_levers()`;
  a `bad_recover` clause in `declared_vs_built` (bank sha256 equals the declared; `p > 0` with no bank, or a bank with `p = 0`, is refused; exclusivity with `--ego-perturb-p` and `--dagger-p`, as :426); a `recover_probe` (the perturbed tensors AND the re-expressed target reach the loss gradient on a real batch; a `p = 0` control reads EXACTLY 0.0; deleting the apply line turns it RED), tests analogous to `tests/test_v8f_ego_perturb.py`.
  ⛔ A flag without a `FLAG_CHECKS` entry makes `self_check` refuse the launch (:576), which is the rule.

## 5. The read (T1-kin; proposed, to be frozen)
Tier T1-kin (self-action OPEN loop, plain causal bucket, nav dead-reckoned, S = sg72 as in FEAS2). Rollouts: R_s and C0_s at s0, s1 (+ C0rep_s), 5-6 x ~150 s at 16 threads. **Read A of R vs P10** (19 rows, D1 / D2 / D3) **and R - C0** (19 rows), all four families per BINDING (longitudinal, lateral, tactical, strategic-PAUSED per ITEM 25).
Primary: M1 crawl share of R - C0 separated-better at both seeds AND >= 0.5 of (C0 - P10)'s M1 excess closed (the HISTSWAP form); guards: ADE6 R - C0 point <= +0.0604 m (F/2) and no separated-worse non-crawl row. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 (answers EPISODES only);
the training-run question needs C0rep (the same-init replicate): INHERITED `LOOP_STATE_2026-11-02.md:8` reports REP - DAG-OACC ADE6 +0.4025 m at s0, above the 0.2415 m bar, so a 4-run design may not be able to read an ADE-sized lever; the M1 floor itself is UNVERIFIED. Inference is deterministic.

## 6. Estimated box-2 wall time
MEASURED per 3,000-step run on box 2 (`summary.json` `seconds`): DAG-OACC 2,275.8 s (s0) / 2,262.7 s (s1); ORC 2,257.8 s vs ORC0 2,249.1 s; DAG-ORC 2,286.3 / 2,293.5 s vs REP 2,240.0 / 2,236.7 s. The perturbation costs <= 2 % (a new bank loader's cost is UNVERIFIED).
**4 runs (R, C0 x 2 seeds) ~ 2.5 h; 6 runs (with C0rep) ~ 3.8 h**, sequential on one GPU, plus the launch gate (CPU, minutes, UNVERIFIED) and the dev-box T1-kin read (~15-25 min for rollouts + read). Zero-GPU build work first (new perturber, flags, probes, tests): UNVERIFIED, plausibly one agent-day.

## 7. The 3 biggest unknowns
1. **Why would this succeed where ORC and DAG-ORC did not?** ORC and DAG-ORC gave the model slowed histories with unchanged targets; DAG-ORC did not clear the crawl rows (ORC's own M1 row was not read here). If H is true, the perturbation law must have been wrong (magnitude, band, or no re-expressed target); the empirical bank is only a guess at which. HISTSWAP says whether the history channel matters, not why ORC missed it.
2. **HISTSWAP's positive reading is leak-confounded.** The logged history carries the human's realised motion at ticks 1 and 2, so SUPPORTED does not separate "own slow history" from "any human-motion history"; a leak-free swap arm does not exist yet. The fine-tune's own read would then be the first discriminating test, at 2.5-3.8 box-2 hours.
3. **Readability.** The measured same-init replicate floor on this rig (+0.4025 m ADE6, INHERITED) can exceed the effects of interest; M1's run-to-run floor is unmeasured, so the 4-run design may be unreadable and the 6-run design is the honest minimum. Also UNVERIFIED: the rejoin blend and its horizon, whether lateral / heading offsets matter for a longitudinal M1, and the bank's coverage of the 3-8 m/s band where the excess lives (H-GAPDECOMP-M-1, INHERITED).
