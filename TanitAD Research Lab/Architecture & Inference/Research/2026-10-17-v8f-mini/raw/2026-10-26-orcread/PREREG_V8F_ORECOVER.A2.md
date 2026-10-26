# PREREG_V8F_ORECOVER: does training on perturbed ego states make the hierarchy's decisions survive the model's own motion?

**Written 2026-10-26, before the lever's code is landed and before any O-RECOVER run exists.** Claim row `H-V8F-ORECOVER-1`.

**PI go (2026-10-26, in chat, verbatim):** *"Yes: queue O-RECOVER on box 2 after OACC and MIXO, through the launch gate, both seeds, under its own
pre-registration."*

## Why (MEASURED)

* `RESULT_V8F_T1_HEADS` (T1-kin, HACC + causal bucket): on its own executed state, **the T head's decisions DEGRADE.**
  * Lateral macro-F1 falls from 0.85 to 0.61 / 0.63, longitudinal from 0.46 to 0.34 / 0.36. The turn classes halve.
  * The drop holds at zero row offset between the labels and the cameras T saw. So it is T reacting to its own ego state, not the replayed
    cameras.
* `RESULT_V8F_FLAT`: O gains nothing over a flat control from S's and T's own outputs (−0.036 / −0.003 m). The GT code would give it 1.2 m.
* `raw/2026-10-26-lcdiag/DIAGNOSIS.md` (`R76`): the loop crawls in launches, where O's own slow pull-away carries part of the deceleration.
* ⇒ The model was trained only on logged ego states. This lever trains it on states off the logged trajectory, toward the human's labels and
  future. That is the DART / ChauffeurNet idea, in its cheapest form.

## 1. The arms (box 2; through `launch_gate.py`; both seeds)

| arm | what | role |
|---|---|---|
| **ORC** | M2b_s{seed} (`D:/tanitad/runs/v8f_mini_full/M2b_s{seed}/ckpt_last.pt`, sha256 `601cb435…` / `6104fb15…`) fine-tuned for **3,000 steps** with the ego perturbation ON | **decisive** |
| **ORC-0** | the same fine-tune with the perturbation OFF (`--ego-perturb-p 0`): same init, steps, data, LR path and seed | **the control.** It separates the perturbation from "3,000 more steps" |

**The perturbation**, as built behind default-off flags; its implementation is landed and gated before any run:
* `--ego-perturb-p 0.3`: the fraction of training samples perturbed.
* `--ego-perturb-vscale 0.5 1.0`, `--ego-perturb-adrop 1.5` m/s², `--ego-perturb-vmax 8.0` m/s: the longitudinal component, exactly
  `PREREG_V8F_HACC_MC` §7.
* `--ego-perturb-yawrate-std 2.0` deg/s: the lateral component.
  * 2.0 deg/s is the scale of the measured T1-kin yaw-rate excess at speed (L3 +1.98 deg/s at s0, `H-V8F-T1-LATCOND-1`). It is fixed now and
    never tuned. The motivation is T's lateral F1 drop. `R76` places most of the curvature excess in the launch crawl, so the lateral
    component is not claimed to fix a lateral deficit.
  * Every ego input the model reads is rebuilt consistently from the perturbed state. **Targets and every S / T label are unchanged.**
* ⚠️ **One lever with two components.** A separated effect is attributed to the perturbation as a whole, never to its lateral or longitudinal
  half. Splitting them is a follow-up arm under its own read.

Everything else is M2b's argv.
* The checkpoint records the built values (`built_levers`).
* The launch gate must PASS for each run: G-DVB on every new flag, and G-LIVE with a non-zero gradient on a perturbed batch.
* ORC-0's 3-step loss trace must equal the default-off trace bit for bit. That is the identity, already pinned by the lever's test.

## 2. Evaluation

| where | what |
|---|---|
| **dev-box CPU (decisive)** | `v8f_t1_eval run --replan-steps 10 --limit causal --dump-ticks --dump-heads` on eval-141, **plain** (no post-map, so no HACC re-fit confound), for ORC, ORC-0 and M2b, × 2 seeds. Read by the T1-HEADS reader (`raw/2026-10-26-t1hdread/t1hd_read.py`), unchanged, plus a pairing step: arm − ORC-0 per seed |
| box 2 | the chain's standard T0 eval (eval-141) of every checkpoint, for the T0 guard and the four families |

⚠️ **One deliberate change from `PREREG_V8F_HACC_MC` §7, which named HACC (re-fitted) and MC-H as the eval configurations.** Here the decisive
read is plain.
* ORC − ORC-0 compares two networks. A post-map re-fitted per arm would add a fitted stage that differs between the arms.
* HACC re-fitted on TRAIN by HACC's rule, for each arm, is a beside read, run only after the decisive one.

**Rows** (paired episode-cluster bootstrap, B = 2,000, per seed, ORC − ORC-0, on the same windows and matched cells):

| row | definition | direction |
|---|---|---|
| **R-TLAT / R-TLON** | the T head's lat / lon macro-F1 at ticks ≥ 1 (the T1-HEADS definition and reducer) | better if > 0 |
| **R-ADE** | T1-kin plain ADE6 | guard: not separated-worse |
| **R-T0** | T0 own-code ADE6 | guard: ORC − ORC-0 ≤ +F/2 = 0.0604 m (point) |

Beside, never decisive:
* S-POS@30 / 50 / 80 against CTRV;
* the crawl fraction M1 and L2;
* the four families at T0 and T1;
* ORC − M2b;
* the tick-0 F1, which shows whether ORC lost logged-state skill.

The seed floor for the F1 rows is the T0 fresh values: lat 0.0124, lon 0.0073, so 2F = 0.0248 / 0.0146.

## 3. Controls (each must pass, or the read is REFUSED)

* The T1-HEADS reader's K1–K7 on every run, as registered there. K4 is replaced for the fine-tuned arms by the identity "tick 0 equals the arm's
  own T0 eval to 1e-3".
* **Launch-gate PASS tokens** for all four runs, with the commit, argv sha256 and data manifest bound.
* **The perturbation reached training:** the run's totals `ego_perturb_n` / `ego_perturb_eligible` (eligible = logged v0 < 8 m/s) are
  written by the trainer, a build item landed with the flag before any run.
  * ORC: the ratio lies in 0.3 ± 0.02. ESTIMATED binomial 3σ: about 0.014 at 10,000 eligible samples.
  * ORC-0: `ego_perturb_n` is exactly 0.
* **Deliberate regression:** already proven at build time (deleting the perturbation line turns the lever's tests RED). No GPU arm is spent on it.

## 4. Outcomes, committed now (ORC vs ORC-0)

| outcome | condition | committed next |
|---|---|---|
| **ROBUST** | R-TLAT and R-TLON separated-better **and ≥ 2F**, at both seeds; R-ADE not separated-worse; R-T0 within +F/2 | adopt ORC as the base. Re-read T1 dominance over P10 and the FLAT control on it (FLAT's committed repeat) |
| **PARTIAL** | at least one F1 row separated-better and ≥ 2F at both seeds, the other not, with the guards held | adopt for the head that moved. The next arm targets the other head's inputs, under its own read |
| **NO EFFECT** | no F1 row separated-better at both seeds, guards held | perturbing the inputs does not teach own-state decisions. Next: on-policy states, i.e. the model's own T1-kin states supervised with the human's labels at the matched route point (DAgger form), under its own PREREG |
| **HARMS** | R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2 | ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1 |
| **REFUSED** | any control fails | fix and re-run |

## 5. Queue

Box 2 runs OACC (now), then MIXO, then **O-RECOVER**: ORC s0, ORC-0 s0, ORC s1, ORC-0 s1.
* Each runs through the launch gate.
* The chain's predecessor predicate reads MIXO's end marker.
* The T1 evals run on the dev-box CPU as each pair of checkpoints arrives, gated on C: ≥ 2 GB free.

## Amendment 1 (`A1-2026-10-26`), before the lever's code and before any run: the fine-tune mechanics fixed

**When.** Written 2026-10-26, about 30 minutes after the PREREG landed (`7ff43701`). No O-RECOVER code is landed and no O-RECOVER run exists.

**Why.** §1 says "fine-tuned for 3,000 steps" without saying how. `stack/scripts/v8f_mini_train.py` has no way to start from another run's
checkpoint, only `--resume` from its own output directory. Its schedule (`lr_at`, `M.p_gt_schedule`) is laid over the run's own step budget, so
a fine-tune would otherwise restart M2b's teacher forcing at p_gt 1.0.

**Fixed now, for ORC and ORC-0 alike.**
* `--init-ckpt <M2b_s{seed}/ckpt_last.pt>`: a new default-off lever, built with the perturbation flags.
  * It loads the model weights strictly into the freshly built M2b. The optimizer is fresh and the step is 0.
  * The path and its sha256 are recorded in `built_levers`, with a G-DVB entry.
  * An existing `ckpt_last.pt` in the run's own output still resumes, as today.
* `--steps 3000 --lr 1e-4 --warmup 100`. The trainer's cosine then runs 1e-4 → 1e-5 over the 3,000 steps.
  * ESTIMATED choice, not tuned on any read: M2b ended at 3e-5 (0.1 × 3e-4). A peak of 1e-4 is about 1/3 of M2b's.
* `--p-gt-start 0.2 --p-gt-end 0.2`: O's teacher forcing is held at M2b's final value, so the fine-tune does not re-open the schedule.
* `--ego-perturb-p 0.3` (ORC) or `--ego-perturb-p 0` (ORC-0) is always NAMED in the argv, because the flag's default disables it.
* `--seed` is the seed of the M2b checkpoint being fine-tuned.

**Disclosed consequence.** ORC-0 is not M2b: it has 3,000 more steps under a different LR and p_gt path.
* That is exactly why ORC-0 is the control.
* ORC − M2b stays a beside row, and is never decisive.

Nothing else moves: the arms, the rows, F, the controls, the outcome table and the committed next steps.

## Amendment 2 (`A2-2026-10-26`), before any run: three defects the reader exposed, fixed before data

**When.** Written 2026-10-26, from the reader's build on SYNTHETIC inputs and a wiring smoke on the banked T1-HEADS runs
(`raw/2026-10-26-orcread/`). No O-RECOVER run exists. No ORC or ORC-0 checkpoint, rollout or eval exists.

1. **K4 cannot pass on a `--limit causal` run, by construction.**
   * Tick 0 of a causal run reads the past-only speed bucket, while the arm's T0 eval reads the corpus's. MEASURED on the banked T1-HEADS M2b
     runs (`out/smoke_real.out`): the causal run's tick 0 differs from its T0 eval by lon F1 0.045 and lat 0.0075; the logged run's by 0.0.
   * T1-HEADS ran K4 on a `--limit logged` run for exactly this reason.
   * **Fix:** each of ORC and ORC-0, at both seeds, gets one more T1-kin run with `--limit logged`, otherwise identical, used ONLY for K4.
     That is 4 extra dev-box CPU rollouts.
   * The decisive rows still come from the causal runs, unchanged.
2. **The §4 table had a hole.** A head separated-better at both seeds, but below 2F at one of them, matched no row.
   * **NO EFFECT now reads:** no F1 row separated-better and ≥ 2F at both seeds, guards held.
   * Disclosed consequence: a head that is separated-better but under the bar is NO EFFECT. Its rows are reported, and it is never called an
     effect.
   * With this, the table is complete. Guards held gives ROBUST, PARTIAL or NO EFFECT by how many heads cleared the bar; a failed guard gives
     HARMS; a failed control gives REFUSED.
3. **K7 had no reference for a plain arm.** "The run is the banked run" referred to T1-HEADS' banked HACC dumps; no O-RECOVER dump is banked.
   * **New K7:** the executed track rebuilt from the dumped plans equals the dumped track and tick poses to ≤ 1e-9 m, per run. It reads 0.0
     on the banked runs.

**How the build meets §1's "every ego input is rebuilt consistently", stated so it cannot be read two ways:**
* `p10f`, `kin3`, `p10_t`, `base`, `pose`, `pose_t` and `pose_s` are rebuilt from one perturbed state: v' = s·v, a' = a − d and
  ω' = s·ω + n per history frame.
* `nav`'s distance to the next manoeuvre is rebuilt like the pose history, scaled by s behind NOW. Its frame-to-frame differences would
  otherwise carry the logged speed. A history frame across a manoeuvre boundary cannot be scaled, so it stays logged and is counted.
* Kept on purpose: the scene, the corpus speed bucket `limit` (an input M2b is trained on, unchanged here), the hold ages, `pose_now`, and
  every target and label.

**Which launch gate.** For this trainer, §1's "through `launch_gate.py`" means its own in-process gate, as for every v8F-mini arm.
* That gate is `v8f_mini_train.py --selfcheck-only`: G-DVB (`FLAG_CHECKS` / `declared_vs_built`, now with entries for the perturbation
  flags, `--init-ckpt` and `--warmup`) and the G-LIVE probes, including a perturbed batch.
* `stack/scripts/launch_gate.py` is the refcv8 gate and never reads this trainer.
* The box-2 chain runs the gate once per run, four times in all. Each writes a PASS token binding the commit, the sha256 of that run's
  training argv, and the data manifest.

**Also fixed here, so the read cannot drift:**
* **ORC − M2b is read at T0 only**, from the eval JSONs, so no M2b T1-kin run is needed.
* **The STRATEGIC family is NOT paused for v8F** (PI ruling 2026-10-17). S-POS against CTRV is reported beside, per run, and decides nothing.
* **The reader's interpretations are adopted as written:** `raw/2026-10-26-orcread/NOTES.md`, items 1–3 and 7–10. Among them:
  * the F1 rows score only cells scored in both runs;
  * bars are inclusive;
  * ORC's and ORC-0's argv must be identical except `--ego-perturb-p` and `--out`;
  * the launch-gate token is checked for its binding to the argv and data, but its HMAC is not verified, because the key lives on the
    launch host.

Nothing else moves: the arms, the decisive rows, F, the bars, ROBUST, PARTIAL and HARMS, and every committed next step.
