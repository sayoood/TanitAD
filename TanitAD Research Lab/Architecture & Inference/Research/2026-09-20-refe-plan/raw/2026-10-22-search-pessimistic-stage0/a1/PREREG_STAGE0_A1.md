# AMENDMENT A1 (SEALED) to the sealed Stage 0 pre-registration: seed-averaged arms

**Status: SEALED by the Master Mind, 2026-10-22 17:00 Berlin, after the calibration and BEFORE any real A1 read.** The body below is the committed draft `DRAFT_AMENDMENT_A1.md` (sha256 `7d6ca9f4…`, recorded in `raw/launch_record.txt` before the calibration was launched), unchanged except this line, the title, the code binding and §9.
**Amends:** `PREREG_STAGE0.md` (sealed 2026-10-22 14:34, sha256 `9cf8980d5cdc6133c72b6944975b992428b97e656db3245925aed48844675de9`, untouched in this directory).
**Evidence class of everything in this amendment:** MEASURED (ours), SELECTION tier, REAL features with SYNTHETIC labels (the calibration) or synthetic worlds (the tests). No real label has been scored by A1, and `raw/RESULT_STAGE0.json` (the withheld real-token gaps of the first read) has not been opened, copied or read.
**Code binding (SEALED, S = 3):** `constants_sha256: cb065815e89487724277ebbd8c767678641a514baa3a4f0fec8cb608e44e3a01` (equal to `raw/constants_digest_by_S.json` entry "3"). `code/stage0.py` sha256 `4eb4ca6b06d0b94d9aad2ae5fe4c279cda85b37c4cac259ef766be8deb564441`: the calibrated file (`d6116a1a…`) with line 49 set to `N_SEEDS = 3` and nothing else.

## 1. Why: the first sealed read was INVALID on C-II-b, and the cause is the noise floor

The first sealed read failed one control, **C-II-b**, the planted-session positive control (`raw/RESULT_STAGE0_controls_only.json`, `controls.C-II-b.per_kappa`; planted labels on the real features, so no real label is in this table):

| κ | planted M1 gap | `lo` (Bonferroni) | swap-null ν | `lo` − ν | SEEN − OOF | exclusion deleted |
|---|---|---|---|---|---|---|
| 0.5 | 0.0141 | 0.0018 | 0.0054 | −0.0036 | +0.0156 | 0.0141 → 0.0010 |
| 1 | 0.0191 | 0.0069 | 0.0102 | −0.0033 | +0.0244 | 0.0191 → 0.0016 |
| 2 | 0.0251 | 0.0065 | 0.0093 | −0.0028 | +0.0330 | 0.0251 → 0.0012 |
| 4 | 0.0348 | 0.0043 | 0.0082 | −0.0039 | +0.0261 | 0.0348 → 0.0044 |

The instrument **sees** the planted effect: the gap rises with κ, SEEN beats OOF at every κ, and deleting the exclusion line collapses it. What fails is the conjunct `lo > ν`. Two IN models scoring the same tokens differ by ν = 0.005 to 0.010 (training noise from fold composition and the minibatch seed), and the Bonferroni lower bound over 21 session clusters is only 0.002 to 0.007. The sealed text's listed cause ("the recipe is too constrained to absorb a session") is **not** what failed. The synthetic power simulation (§8 of the sealed text) under-stated both ν and the interval.

**The next arm (RULE ZERO).** ν is model-to-model variance. Averaging S models that differ only in the minibatch seed divides the seed part of that variance by about S, for every role at once, so the comparison stays symmetric. A1 does exactly that and nothing else.

## 2. What A1 changes (this list is complete)

1. **Seed averaging.** Every fold model *h* is the **mean of the composite scores of S models** that are identical except for the minibatch seed. Seeds: the primary arm trains seeds `0 .. S−1`; the training-seed replicate (sealed §5) trains the disjoint seeds `100 .. 100+S−1`. The average is over the **composite score** `σ(NC)·σ(DAC)·(5σ(EP)+5σ(TTC)+2σ(C))/12`, never over weights (the composite is nonlinear in the logits, and `test_A1_a_fold_model_scores_the_mean...` pins that the two readings differ on the test world).
2. **Every role is averaged:** OOF, IN, all swap-null models, the replicate, the descriptive R1 arm, the exclusion-deleted C-III arm, the planted and the exclusion-deleted planted arms of C-II-b, and the heads trained on shuffled labels in C-I (trained). Each seed is one unamended fit, so the exclusion line and its fail-closed structural check run for every one of the 5·S models before any optimiser step.
3. **Constants.** `N_SEEDS = S` and `REPL_SEED_BASE = 100` enter the digested constants block (`TRAIN_SEEDS = (0, 1)` is replaced by the derived `primary_seeds` / `replicate_seeds`); the schema string becomes `spess0-a1`. `seed_sets()` fails closed if the two seed sets ever overlap.
4. **Consequence at S = 1.** The pipeline is the unamended one **bit for bit** (§5), with one named exception: the replicate arm uses seed **100** where the sealed code used seed 1. Same pipeline, a different draw of the seed; nothing else differs.

## 3. What A1 does NOT change

Folds, halves, the recipe R0/R1 and its 600 steps, the three measures, Δ = (0.01, 0.02, 0.35), α = 0.025 with Bonferroni over three, the cluster bootstrap (B = 10,000, seed 20261022), the model-swap null, leave-one-session-out, the decision rule `decide()` (the four conjuncts, GAP through any one certified measure, the POWERED / UNDERPOWERED labels), and every control **with its bound** (C-STRUCT, C-I-eval, C-I-trained, C-II-a as non-reversal, C-II-b on any planted κ with the 0.02 minimum, C-III-real gating on M1). The planted generator (seed 7771, sharpness 1) is unchanged. The sealed §11 judgement calls stay as confirmed.

## 4. The calibration and the SELECTION RULE (committed before the calibration was launched)

`code/calib_a1.py`, REAL features with SYNTHETIC labels only (the real labels are replaced by ones at load, `assert_blind` is checked in every worker process, and a test proves the guard goes RED on a world that carries labels). For each S ∈ {1, 3, 5, 8}:

* the **four planted worlds** κ ∈ {0.5, 1, 2, 4}, generator seed 7771, exactly the sealed run's C-II-b arms (the planted arm and its exclusion-deleted twin, both S-averaged);
* **24 κ = 0 null worlds** (the same generator at κ = 0, label seeds 8001 to 8024, same real features), each through the full decision machinery: primary arm, seed-replicate arm, exclusion-deleted arm, C-I evaluation (×100) and trained (×10), C-II-a, C-III-real, and `decide()`. The descriptive R1 arm never enters `decide()` and is not run in the calibration (it is run in the real read).

C-II-b depends on no label (the planted worlds are built from the features alone), so it is one event per S, computed once and joined to every null world. Definitions:

* **False positive (FP)** = the full rule reads **GAP** on a κ = 0 world, with C-II-b taken as passed. (Taking it as passed keeps a failed C-II-b from hiding false positives behind INVALID.)
* **Null-world INVALID** = any control **other than C-II-b** fails. (A failed C-II-b would otherwise make 24 of 24 worlds INVALID: one event counted 24 times. It is gated by condition (a) instead.) `invalid_with_c2b_n` is also recorded.

> **Selection rule.** Choose the **smallest S in {3, 5, 8}** such that
> (a) C-II-b passes at some kappa <= 2 (i.e. κ ∈ {0.5, 1, 2}; the sealed C-II-b rule: M1 gap ≥ 0.02, `lo` > ν, SEEN − OOF > 0 in the planted world, and the gap collapses when the exclusion is deleted);
> (b) the null-world false-positive count is <= 2/24;
> (c) the null-world INVALID count (excluding C-II-b) is <= 4/24.
>
> **If no S qualifies, the outcome is INSTRUMENT-LIMITED:** no real read. The route is then `D-SESSION-RESERVE` (PI) or the decoder-level variant (~1.5 GPU-h).

An S whose calibration is incomplete (any task errored) cannot qualify. S = 1 is the identity check and the baseline and is never selectable. `calib_a1.py` implements the rule once (`apply_selection_rule`, tested on its boundaries: 2/24 passes and 3/24 fails; 4/24 passes and 5/24 fails; κ = 4 alone does not satisfy (a)) and prints its verdict; it does not edit any constant. The calibration writes `raw/calib_a1.json` and prints `ZZCALIB_DONE` as its last line.

**Reading the rule.** (b) is looser than the nominal α = 0.025 (0.6 of 24) on purpose: it is a screen against a rule that manufactures GAPs under seed-averaged training, not a power claim. (c) bounds how often the other controls void a valid read. Neither number was tuned on a calibration output.

## 5. Tests and the S = 1 identity (MEASURED, `raw/`)

* `golden_digests.py` was run against the **sealed** `stage0.py` (sha256 `71a9ccf5…`) before the amendment existed (`raw/golden_unamended.json`); the amended code at S = 1 returns the **identical** digests for the arm statistics and for the raw bytes of every cached composite score, on a null and a planted synthetic world, the exclusion-deleted arm and the trained-on-shuffled-labels control. Pinned in `test_stage0.py` (`GOLDEN_UNAMENDED`).
* On the **real features**, S = 1, the four planted worlds reproduce the sealed run's C-II-b numbers (M1 gap, `lo`, ν, SEEN − OOF, the collapsed gap and its ν) with **max |diff| = 0.0, bit identical** (`raw/preflight_identity.json`; the calibration repeats and records it in `raw/calib_a1.json`).
* The test results and the mutation census on the changed lines are in `raw/pytest_final.txt` and `raw/MUTATION_CENSUS_A1.json` (see HANDOFF.md).

## 6. ESTIMATED runtime

Basis: MEASURED per-phase times on the real features, one process, S = 1, uncontended (arm ≈ 3.5 s, C-I evaluation ×100 ≈ 22 s fixed, C-I trained ×10 ≈ 33 s, one planted pair ≈ 7.6 s), scaled linearly in S for everything that trains a model; a pre-flight with 8 worker processes on a loaded box ran about 1.7 to 2.3× slower.

| S | real read, serial, uncontended (ESTIMATED) | same on a loaded box | + the test suite that runs first |
|---|---|---|---|
| 1 | about 2 min | about 4 min | about 1.5 min |
| 3 | about 4.5 min | about 9 min (pre-flight: 558 s) | about 1.5 min |
| 5 | about 7 min | about 14 min | about 1.5 min |
| 8 | about 11 min | about 22 min | about 1.5 min |

Peak memory of the real read is under 1 GB (the calibration worker peaks at 0.53 GB with the label sets streamed). **The calibration itself** (112 tasks, 20,000 CPU-s uncontended ESTIMATED; at the 1.7 to 2.3× slowdown seen in the pre-flight, 34,000 to 46,000 CPU-s) is launched with **12 worker processes** (the brief allows 20; the dev box is shared, it had 5 GB of free RAM at launch, and a worker peaks at 0.53 GB): **ESTIMATED 50 to 65 minutes wall**, up to about 90 on a heavily loaded box. `calib_a1.py` prints one progress line per finished task.

## 7. Disclosures and UNVERIFIED items

1. **A pre-flight ran before this text was written.** To check plumbing and time the pipeline I ran (i) S = 1, the four planted worlds (the identity check, §5) and (ii) S ∈ {1, 3} with 2 null worlds. These used synthetic labels only. The selection rule in §4 is the one given in the build brief and was not changed after the pre-flight; the pre-flight showed C-II-b passing at κ = 2 for S = 3 and failing for S = 1, which is the direction the amendment predicts, and **is not the calibration** (2 null worlds cannot estimate a false-positive rate). The calibration below is the evidence.
2. **The null worlds are generated by the C-II-b generator at κ = 0** (P2's own logit, Bernoulli labels). They test the rule's false-positive behaviour under that generator's noise; they are not REFe's label noise. A GAP-rate of zero here does not bound the real false-positive rate.
3. **A pass of C-II-b at S proves the instrument can see a planted effect of that size under averaging; it does not say the real session effect is that large.** The real read's NULL is still conservative-only evidence, per sealed §9.
4. The real read at the chosen S has not been run. `raw/RESULT_STAGE0.json` (first read, real gaps) remains withheld on the dev box in `spess0/raw/`.
5. `geometry_real.json` in `spess0/code/` does not match the old `MANIFEST.sha256` line (recorded at the seal); it feeds only the synthetic worlds. The copy here is byte-identical to `spess0/code/geometry_real.json`.

## 8. After the calibration (the MM)

Apply the rule to `raw/calib_a1.json` (`selection.verdict`), set `N_SEEDS` in `code/stage0.py`, re-digest (`python code/stage0.py constants`), quote that digest here, finalise this text as `PREREG_STAGE0_A1.md` with the chosen S written into §2, seal it (`sha256sum` into `PREREG_STAGE0_A1.sha256`), and run `run_stage0.cmd` / `.sh` (they now read `PREREG_STAGE0_A1.*` and write `raw/RESULT_STAGE0_A1.json`). The register row `D-REFE-SESSION-EXPOSURE-1` is updated in the same turn as the read. If the verdict is INSTRUMENT-LIMITED there is no read.

## 9. Calibration result and the selection (MM, at seal, 2026-10-22 17:00 Berlin)

`raw/calib_a1.json`: `n_errors` 0; identity at S = 1 PASS, reproducing the sealed run's C-II-b numbers bit for bit; wall 6,234 s with 12 workers. The rule's verdict is **`SELECT S=3`**, checked by hand against the per-S table:

| S | C-II-b κ read (≤ 2 required) | null-world GAP (FP, ≤ 2/24) | null-world INVALID excl. C-II-b (≤ 4/24) | qualifies |
|---|---|---|---|---|
| 1 | none (the identity: reproduces the INVALID) | — | — | never selectable |
| **3** | **κ = 2** | **0 / 24** | **0 / 24** | **yes: the smallest** |
| 5 | κ = 2 | 0 / 24 | 1 / 24 | yes |
| 8 | κ = 2 | 0 / 24 | 2 / 24 | yes |

**What this licenses, stated before the read:**
- At S = 3 the instrument reads a planted session effect at κ = 2 (realised M1 gap about 0.025 in the sealed run), not at κ ≤ 1.
- So a NULL can only be labelled POWERED if `hi_1 < Δ_1` and `hi_3 < Δ_3` hold on the real tokens (§7 of the sealed prereg, unchanged).
- Otherwise it is NULL-UNDERPOWERED, which licenses nothing (the routing in §7 of the sealed prereg is unchanged).

**Seal block:** sealed 2026-10-22 17:00 Berlin by the MM. This file's sha256 is in `PREREG_STAGE0_A1.sha256`. The withheld first-read file `C:/Users/Admin/par18/spess0/raw/RESULT_STAGE0.json` is still unopened by the MM and the A1 agent. It is banked once this read concludes.
