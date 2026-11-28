# PREREG_V8F_T1KIN_STRICT: HACC at T1-kin with BOTH open channels closed (cameras frozen at t0, causal speed bucket) (2026-10-26)

**Written before any number of this configuration exists.** Master Mind, 2026-10-26 ~01:55 Berlin. Claim row `H-V8F-T1KIN-STRICT-1`. Zero training.

## 0. Why (MEASURED)

`RESULT_V8F_T1KIN_HACC` and `RESULT_V8F_T1KIN_CAUSALBUCKET` give HACC − P10 ADE6 at T1-kin m = 10 as follows:

| configuration | s0 (m) | s1 (m) |
|---|---|---|
| default (cameras logged, oracle bucket) | −0.829 | −0.795 |
| cameras frozen at t0 (oracle bucket) | −0.787 | −0.706 |
| causal bucket (cameras logged) | −0.768 | −0.734 |

Each channel was closed alone. Closing both at once is the strictest T1 the programme can run without re-perception (T2 is not provisioned). In that configuration the model reads only:
* the first frame;
* its own executed motion;
* a speed bucket from the speed it has already driven;
* the dead-reckoned nav.

The floor reads neither channel, so this is the cleanest model-vs-floor comparison at T1.

## 1. The run

`v8f_t1_eval.py` (tip `76c7d98e`) with `--replan-steps 10 --nav deadreckon --scene frozen --limit causal --schedule periodic`.
* HACC checkpoints `M2b_s{0,1}_hacc.pt` (the sha256s of `t1k_chain.sh`).
* eval-141. Dev-box CPU.
* Every run asserts the tool's sha256 before and after.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, with Amendment 1's rows read through `t1k_read.py`'s functions. F = 0.1207 m, 2F = 0.2415 m.

## 2. Controls (a failure ⇒ REFUSED)

* **The floor is untouched:** P10's executed tracks equal the T1KIN m = 10 floor's to ≤ 1e-9 m (it reads no camera and no bucket).
* **Both channels are closed:**
  * The tick-0 plans equal the causal-bucket run's tick-0 plans to ≤ 1e-6 m, because at tick 0 the frozen and the logged frame are the same frame.
  * The executed tracks differ from the causal run's by > 1e-3 m in at least one window.

## 3. Outcome (no bar moves after data)

**PRIMARY**, per seed: HACC(strict) − P10 ADE6 point ≤ −2F, with the interval excluding 0.

| outcome | condition | next |
|---|---|---|
| **STRICT MARGIN** | PRIMARY at both seeds | v8F's T1 headline is stated in the strict configuration (both open channels closed) |
| **STRICT PARTIAL** | separated-better at both seeds, PRIMARY failing at one or both | Report the margin per seed. The headline stays on the causal-bucket configuration, with the strict number beside it |
| **STRICT LOST** | separated-better at neither seed | The margin needs the live camera stream together with the speed channel. Stated as such |
| **MIXED** | anything else | per seed |

**Beside:** the four-family rows and DOMINANCE; strict − causal-bucket on every row.

⛔ Not closed loop and not "drives": the first frame is the human's, and nothing is re-rendered.
