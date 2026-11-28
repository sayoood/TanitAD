# PREREG_V8F_T1KIN_CAUSALBUCKET: does HACC's T1-kin margin survive a speed bucket built only from the past? (2026-10-26)

**Written before any number with the causal bucket exists.** Master Mind, 2026-10-26 ~01:40 Berlin. Claim row `H-V8F-CAUSALBUCKET-1`. Zero training.

## 0. Why (MEASURED, `RESULT_V8F_T1KIN_HACC.md`, `96604184`)

* **HACC − P10 at T1-kin m = 10:** ADE6 −0.829 / −0.795 m, separated at both seeds (outcome T1 BEATS THE FLOOR).
* **With the speed bucket removed** (`--limit off`), it is +0.238 / +0.078 m, not separated. The bucket carries the margin.
* **The bucket's provenance, read from source** (the sidecar's own meta, `refcv6_speed_max_v8_train.jsonl.meta.json`, produced by `stack/scripts/build_refcv6_speed_max_window.py`):
  * It is the CONTAINING WINDOW over {30, 50, 100, 120} km/h, mapping (0, 30] → 30, (30, 50] → 50, (50, 100] → 100, and above 100 → 120 (clamped).
  * Its source is `g_tac.goals.SPEED_BAND.v_hi_ms`, the max of the ego's OWN REALISED speed over [t0 + 2 s, t0 + 6 s]. That is one value per clip, constant over the clip.
  * The PI authorised that future-ego derivation on 2026-09-16, as a declared input.
* **"Off" mixes two things:** the bucket's information, and the model's response to an absent input (it was trained with the bucket present). A bucket that is PRESENT but CAUSAL separates them. It is the deployable form of the input: a set speed known from the past alone.

## 1. The arm

**`--limit causal`** (new mode of `v8f_t1_eval.py`). At every tick j, the bucket the model reads is the containing window, by the sidecar's rule above, of 3.6 · max v over the measured ego speed rows [j − 20, j] (the last 4 s; clipped at the clip start).
* The speed array is the tool's `WindowState.d["v"]`: logged up to t0 and the model's own executed speed after t0. Rows after j are NaN by construction.
* `limit_ok` is True. A window whose 4 s past holds no finite speed gets the logged clip bucket's FORMAT with `limit_ok` False, exactly as `--limit off` builds it, and is counted.
* The bucket can change between ticks, unlike the logged per-clip constant.
* ⛔ No future row and no label enter it. It is not the situation classifier and reads nothing from it.

**Settings: exactly `PREREG_V8F_T1KIN_HACC`'s decisive settings with one flag changed.**
* m = 10, `--nav deadreckon --scene logged --schedule periodic --limit causal`.
* HACC checkpoints `M2b_s{0,1}_hacc.pt` (the sha256s of `t1k_chain.sh`).
* eval-141: 2,399 windows, 141 episodes. Dev-box CPU.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000. Its rows are Amendment 1's four-family rows, read by `raw/2026-10-26-t1kin-hacc/t1k_read.py`'s functions.
* F = 0.1207 m, 2F = 0.2415 m.

## 2. Controls (read before the outcome; a failure ⇒ REFUSED)

* **C-L1:** with `--limit logged` and `--limit off`, the patched tool's output is bit-identical to the unpatched tool's, on a `--max-windows` smoke.
* **C-L2 (causality):** NaN-poisoning, or overwriting with 99 m/s, every speed row after j leaves the causal bucket at tick j unchanged. A deliberate-leak mutant that reads row j + 1 turns a test RED.
* **C-L3:** on 20 windows at tick 0 (logged history), the bucket equals a hand computation from the cache's logged speed.
* **Reachability:** in at least one window the causal bucket differs from the logged one, and the plan there differs from the `--limit logged` plan by > 1e-3 m.
* **The floor is untouched:** P10 reads no bucket, so the floor's ADE6 equals the T1KIN m = 10 floor to ≤ 1e-9 m.

## 3. Read and outcome (no bar moves after data)

**PRIMARY**, per seed: HACC(causal) − P10 ADE6 at T1-kin m = 10, point ≤ −2F, with the interval excluding 0.

| outcome | condition | next (committed) |
|---|---|---|
| **CAUSAL KEEPS THE MARGIN** | PRIMARY holds at both seeds | v8F's T1 result is stated on the causal bucket. No future-derived input enters it; `D-V8F-SPEEDBUCKET-T1` stops gating the headline. HACC-LAT's deployed arm is then re-read with `--limit causal` as a beside run |
| **CAUSAL PARTIAL** | separated-better at both seeds, but PRIMARY fails at one or both | Report the margin per seed. Next: train O with the causal bucket in place of the logged one (box 2, its own PREREG), so that the input it reads at inference is the one it learned on |
| **CAUSAL LOSES IT** | separated-better at neither seed | The T1 margin is the oracle bucket's. Next: the same train-time arm, and the perception-side speed lever (LoRA, `D-V8F-SPEED-PERCEPTION`, A40 at the PI's go) |
| **MIXED** | anything else | Reported per seed |

**Beside (decides nothing):**
* The four-family rows and DOMINANCE of HACC(causal) − P10.
* HACC(causal) − HACC(logged) on every row.
* The share of ticks whose causal bucket equals the logged one, and the confusion between them.
* The causal bucket over the whole past ([clip start, j]) instead of 4 s, as one more run per seed.

## 4. What this does NOT license

* ⛔ Closed-loop or "drives" wording. ⛔ Choosing the 4 s window, the rule or the bar after a number is seen. ⛔ Calling the causal bucket a speed LIMIT: it is a past-speed proxy, and a limit sign is absent from PhysicalAI (`D-MAXSPEED-ABSENT-ALL-LAYERS-1`).
