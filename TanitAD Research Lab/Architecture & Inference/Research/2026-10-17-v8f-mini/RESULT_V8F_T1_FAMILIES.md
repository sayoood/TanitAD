# RESULT_V8F_T1_FAMILIES: at T1-kin, HACC's executed tracks are NO WORSE than P10 on the TACTICAL and DISTANCE-KEEPING families, and separated-better on most rows. The read as written was REFUSED, and this outcome stands under a POST-DATA amendment of one control

`Project Steering/PREREG_V8F_T1_FAMILIES.md` (`635487e5`). Claim row `H-V8F-T1-FAMILIES-1`. Zero training and zero rollout: it re-reads the banked
T1-kin m = 10 dumps.

* **Readers:**
  * `raw/2026-10-26-t1-families/t1kf_read.py`, the PREREG as written (sha256 `c9b49c4a…`);
  * `t1kf_read_a1.py`, the same reader with only C6 replaced (`3072c797…`).
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, seed 0, arm − P10 per seed, 141 episodes.
* **Tier:** T1 (T1-kin: a self-action OPEN loop, never "closed loop").
  * ⚠️ The lead follows its LOGGED future and does not react to the ego.

## 1. The read as written: REFUSED

Controls C1–C5 passed. **C6 FAILED**, so both families were REFUSED. That refusal stays in the record (`T1KF_READ.json`, sha256 `8a0e1452…`).

C6 rolled the decisive HACC tracks by one window and compared them with P10. The rolled tracks read separated-*better* on TAC-LAT (+0.130) and
TAC-LON (+0.112), and n.s. on DK-HW. There are two reasons:

* **The roll is a near-copy.** Grid windows are 0.8 s apart on a 6 s horizon. Rolled − unrolled HACC: TAC-LAT −0.007 n.s.
* **The comparator could not show it.** P10 is a weaker tactical arm than even the rolled HACC: TAC-LAT accuracy 0.593 against 0.723.

The control was written without computing what it tests. That is the class of `R74`, a second time today (N-3 of `PREREG_V8F_T1_LATCOND` was the
first).

## 2. Amendment 1: C6 → C6A, written AFTER the decisive rows were seen

The reader had reported the decisive rows as data beside its refusal, so the amendment came after they were seen. The read under it is **POST-DATA,
not blind**.

* **What did not move:** the rows, the bars, the comparator, the estimator and the outcome table.
* **C6A:** every window is scored with the decisive HACC track of the window from a DIFFERENT episode whose v0 is nearest (median |Δv0| 0.004 m/s,
  1,614 distinct donors). This is the right speed on the wrong road and the wrong lead. It is compared with the unshuffled HACC tracks.
* **It must read:** TAC-LAT, TAC-LON and DK-HW separated-worse at s0.
* **It reads:** all five rows separated-worse, at both seeds.

| C6A: donor − HACC | s0 | s1 |
|---|---|---|
| TAC-LAT | −0.243 [−0.290, −0.198] | −0.244 [−0.294, −0.193] |
| TAC-LON | −0.114 [−0.157, −0.072] | −0.117 [−0.155, −0.080] |
| DK-VIOL | +0.050 [+0.019, +0.089] | +0.040 [+0.014, +0.074] |
| DK-HW (m) | +1.30 [+0.79, +1.92] | +1.26 [+0.82, +1.76] |
| DK-TG (s) | +0.26 [+0.15, +0.40] | +0.23 [+0.15, +0.32] |

⇒ The instrument sees a known-bad arm on every row. C1–C5 and the internal checks X1–X3 pass in this run too.

## 3. The read (decisive: HACC with the causal speed bucket − P10)

* **The LEAD rows:** 743 LEAD windows (595 at risk). Beside them, 552 NO_LEAD, 931 NOT_STRAIGHT and 173 NO_LABEL windows are counted and never scored.
* **The join's label-free speed proof:** max \|v0 − block speed\| = 2.6e-5 m/s.
* **The kinematic contract:** median \|d headway\| = 0.022 m.

| row (worse if) | s0 | s1 | arm means s0 / s1 (P10) |
|---|---|---|---|
| **TAC-LAT** accuracy (< 0) | **+0.136 [+0.101, +0.175] better** | **+0.092 [+0.050, +0.135] better** | 0.730 / 0.685 (0.593) |
| **TAC-LON** accuracy (< 0) | **+0.131 [+0.084, +0.177] better** | **+0.132 [+0.086, +0.174] better** | 0.503 / 0.503 (0.371) |
| **DK-VIOL** (> 0) | −0.013 [−0.038, +0.007] n.s. | −0.008 [−0.029, +0.008] n.s. | 0.034 / 0.039 (0.047) |
| **DK-HW** \|headway − human\| (m) (> 0) | **−0.32 [−0.60, −0.08] better** | −0.27 [−0.56, +0.004] n.s. | 2.13 / 2.15 (2.39) |
| **DK-TG** \|time-gap − human\| (s) (> 0) | **−0.101 [−0.164, −0.050] better** | **−0.073 [−0.124, −0.028] better** | 0.329 / 0.358 (0.423) |

* **The seed floor (H-ESTIM-SEED-1):**

  | row | F | \|delta\| / F, s0 / s1 |
  |---|---|---|
  | TAC-LAT | 0.045 | 3.1 / 2.1 |
  | TAC-LON | 0.0004 | 315 / 316 |
  | DK-HW | 0.024 m | 13.6 / 11.3 |
  | DK-TG | 0.028 s | 3.6 / 2.6 |

  Only TAC-LAT at s1 (2.1×) is close to the 2F reading rule.
* **The beside configurations** (logged bucket; strict: first frame held plus the causal bucket): no row is separated-worse at either seed. Every
  delta points the same way.

**Outcome (verbatim, under Amendment 1): TACTICAL NO WORSE; DISTANCE KEEPING NO WORSE.**

## 4. What this settles, and what it does not

* ⇒ **On the executed T1 tracks, HACC's manoeuvres match the human's more often than P10's.**
  * The lateral 3-way agreement is +9 to +14 points and the longitudinal 3-way agreement +13 points, separated at both seeds.
  * Behind a lead, HACC keeps a headway and time gap closer to the human's.
  * LEAD-VIOL is no worse.
  * The ADE6 margin of `RESULT_V8F_T1KIN_HACC` is therefore not bought with a worse decision or a closer gap.
* ⚠️ **It does not license "dominance" on these families.** No better-bar was registered, and the read is post-data under Amendment 1. The outcome
  words are NO WORSE, and the "better" cells are data.
* ⚠️ **The tactical family here is trajectory-derived.** The canonical factored labeller is applied to the executed track. It says what the car DID,
  not what the T head DECIDED.
* **Not covered (still NOT COMPUTED, work items):**
  * STRATEGIC at T1;
  * the model's own T-code decision quality at T1.

  Both need the per-tick head dump that is being built (a `--dump-heads`-class flag on `v8f_t1_eval`), then a re-run of the decisive configuration.
* **The open gap at T1 is still the lateral one:** curvature at speed (`RESULT_V8F_T1_LATCOND`, CONDITIONED LATERAL DEFICIT). Its root cause is
  being diagnosed.

**Update (2026-10-26, `R76`):** the "lateral" gap of §4 is localised. It is a low-speed launch CRAWL that the GT-speed-masked curvature row prices as lateral; at equal own speed HACC's curvature error equals P10's (`raw/2026-10-26-lcdiag/DIAGNOSIS.md`). The lever is the along weight: `PREREG_V8F_HACC_MC.md`.
