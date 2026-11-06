# FACTS for the v8F paper and the investor pitch video (Master Mind, 2026-11-06)

Every number used in the paper or the video comes from this sheet, with the tier it carries. Sources are repository paths at tip
`24b97ba9`. `R/` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`. The v41 report is
`R/report/2026-11-06-v41/V8F_SCIENTIFIC_REPORT.md` (§ numbers refer to it).

**Wording rules (binding):**
* Never write "closed loop" / "closed-loop" for any number. T1-kin is a self-action open-loop rollout on logged cameras, and T2 is
  re-rendered simulation, in preparation.
* Never call a result "dominant".
* Never claim state of the art or a leaderboard rank.
* State the tier for every result.
* Positive framing is welcome. A claim beyond these numbers is not. Ideas that are not implemented are labelled "outlook" or
  "in preparation".

## The programme (4B), from `Paper/TANITAD_PAPER.md` §3
* **TanitAD's thesis.** A sub-300 M-parameter, self-supervised hierarchical latent world model, the **4B architecture**, made of four
  brains:
  * **operative:** 10–20 Hz, 0.1–2 s, continuous control;
  * **tactical:** 1–2 Hz, 2–6 s, manoeuvre selection by imagine-and-select;
  * **strategic:** ≤ 0.5 Hz, 6–30 s, routing over a latent transition graph without HD maps;
  * **fallback:** continuous self-monitoring that owns the minimal-risk manoeuvre, as the substrate for ISMR / DSSAD.
* **Each level** has its own predictor in its own latent space. Goals condition strictly downward, latents flow upward through stop-gradient,
  and gradient isolation is checked, not assumed. The state shrinks and the clock slows going up (d_S < d_T < d_O, τ_S > τ_T > τ_O).
* **Objective:** residual multi-horizon latent prediction (change-weighted), SIGReg (LeJEPA's sliced Epps–Pulley test, Cramér–Wold),
  inverse dynamics, and imagination of unobserved areas with a calibrated per-cell variance.
* **The 4B flagship** (263.4 M parameters measured, `MODEL_REGISTRY §1.2`) is the programme's first sub-floor arm: ADE@2 s 0.452 m
  against a best-of-3 kinematic floor of 0.500 m and CTRV at 0.523 m. On windows where kinematics diverge from the future it beats
  CTRV by 0.80 m, and the whole advantage is vision (TANITAD_PAPER abstract, v0.4; open loop).
* **Theory cited by the paper:** planning regret grows linearly with the horizon, so per-level short horizons give smaller bounds.
  Hierarchical recombination turns sample complexity from a product into a sum of level cardinalities.

## v8F-mini, the current reference implementation (report §1, §4)
* **Three supervised levels over frozen DINOv3-L/16 features:**
  * S: route checkpoints at 30 / 50 / 80 m, about 1 Hz;
  * T: a typed 60-dimensional tactical code (22 goals, 5 lateral and 7 longitudinal manoeuvre classes, 26 continuous constraints),
    about 2 Hz;
  * O: the 6 s trajectory, 10 Hz.
* **Plus** an H-JEPA world-model auxiliary on the shared adapter.
* **Residual on a frozen learned kinematic prior P10**, in a ramped 12-term cosine basis. A zero-initialised head equals the prior bit for
  bit, so every learned metre is measured against the floor.
* **Whole-code scheduled teacher forcing**, p_GT from 1.0 to 0.2.
* **Size and cost:** 5.2 M (M1) / 7.2 M (M2b) trainable parameters. Trains in about 4 h on one 8 GB consumer GPU.
* **Inputs at inference:** one front-wide camera, measured ego motion including yaw rate (no steering angle), nav, and a speed bucket.
  A 14-key allow-list and a key guard enforce them.
* **Data:** PhysicalAI-AV, 4,512 training clips (306,566 windows). Evaluation: 141 held-out episodes, 2,399 windows.
* **Estimator:** paired episode-cluster bootstrap, B = 2000. It answers "another draw of episodes". Training-seed variance is reported
  separately through two seeds.

## Results (all MEASURED; tier stated)
1. **K-1, open loop with the model's own held outputs.** M2b improves 6 s ADE over P10 by 0.312 / 0.432 m at two seeds, both separated
   and beyond the pre-registered bar 2F = 0.2415 m. That is a 12.5–17.4 % lower 6 s error (§7.1).
2. **HACC, a zero-training post-map, open loop.** It is separated-better than P10 on every row of a pre-registered dominance row at both
   seeds: ADE2 −0.011 / −0.008 m, ADE6 −0.50 / −0.47 m, accel MAE −0.022 / −0.040 m/s², plus curvature, speed, heading, yaw rate,
   decision agreement and consistency. Distance keeping is not separated, and nothing is separated-worse (§7.12). Phrase it as "better
   than the kinematic floor on every row of the pre-registered set, in open loop".
3. **T1-kin, the model driving on its own executed motion, cameras logged.**
   * HACC beats P10 by 0.83 / 0.79 m of 6 s ADE.
   * Rebuilt from the last 4 s of measured speed only (no future information), the margin is 0.77 / 0.73 m.
   * With the first frame frozen for 6 s plus the past-only bucket, it is 0.73 / 0.64 m (§7.15).
4. **The interface carries information (oracle-by-interface, T0 diagnostic).** Feeding O the ground-truth tactical code cuts 6 s ADE from
   about 2.1 m to about 1.0 m: about 1.1 m of headroom through the typed interface (§1, §7.3).
5. **The strategic level, T1-kin.** S's route checkpoints beat a kinematic arc at 30 / 50 / 80 m by 1.2 / 3.3 / 7.8 m and
   1.3 / 3.5 / 8.8 m, separated at both seeds (§1).
6. **DAgger on the model's own states (p = 0.1), T1-kin.** Outcome ROBUST.
   * T's lateral macro-F1 +0.108 / +0.118, longitudinal +0.055 / +0.081.
   * Executed 6 s ADE −0.54 / −0.40 m against its control.
   * The first training lever to clear a T1 bar on both heads (§7.19).
   * The DAgger arm itself beats P10 by 1.17 / 0.99 m of 6 s ADE on its own motion (§7.20; used in the paper's Fig. 3).
7. **The adopted base DAG-OACC against P10, T1-kin.**
   * 6 s ADE −1.2556 / −1.1903 m.
   * Every one of the four families (longitudinal, lateral, tactical, strategic) has a separated-better row (§7.34, Read A D1 and D3).
   * With the feasibility post-step the composite fails only one of 19 rows: the launch-crawl share (`R/raw/2026-11-05-motion-c-ri/ri_read/RESULT_RI_C0.md`).
8. **The remaining frontier, stated positively.** The one open row is the launch crawl: where the human pulls away from near-standstill,
   the model starts slower.
   * Every zero-training CPU lever has been measured.
   * The exact human speed, as a ceiling diagnostic, closes up to about half of it with a bounded floor (§7.41).
   * The programme's analysis attributes the rest to what logged cameras can show. That is why re-rendered simulation (T2) is being
     built.

## In preparation (built, not yet run)
* **T2 re-perception rollouts on NuRec neural reconstructions.**
  * 62 scenes (42 launch, 20 guard) selected by a rule frozen before any log was read: 35.7 GB, staged.
  * The harness reproduces the T1-kin plans bit for bit from cached features, and the in-loop encoder reproduces the cached tokens to
    2.6 % rel-L2 (cosine 0.99965).
  * The pre-registered card commits both outcomes in advance.

## Outlook (not implemented; label it so)
* **Perception:** a LoRA fine-tune of the visual encoder for motion cues. It needs an A40-class GPU, about 100 GPU-h per seed (ESTIMATED).
* **Scale:** data (the PhysicalAI-AV corpus is far larger than the 4,512 clips used) and size (from 7 M toward the 300 M 4B budget).
* **Strategic topology:** routing from a latent transition graph, with map structure from simulation or an external corpus.
* **The fallback brain:** imagination-error self-monitoring and the minimal-risk manoeuvre (ISMR / DSSAD logging).
* **Imagination of occluded areas:** multi-step calibrated belief rollout.
* **Deployment:** Jetson Thor, at most 100 ms end to end with 4 cameras (PI requirement; optimisation follows results).
* **Benchmarks:** NavSim / navtest and T2 dominance reads.

## Method rigour (a selling point)
* Every arm is pre-registered with both outcomes committed before data.
* A launch gate binds every training run to its exact commit, argv and data hashes.
* Deliberate-regression arms prove that every check can fail.
* There are three variance questions (episodes, training seeds, inference), and each claim names which one its interval answers.
* Four metric families are reported on every eval, never ADE alone.
* There is an append-only retraction log.
