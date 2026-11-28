<title>navhard clause-stratum confirmation</title>

# `D-NAVSIM-CLAUSE-STRATUM-1` — the confirmatory replication of "the clause is an amplifier, not the cause", pre-registered on navhard before its scores exist

`Benchmarks & Evals · 2026-09-20 · Master Mind · PRE-REGISTRATION — no navhard score seen`

## 0 · Why this is a pre-registration and not a repeat

The warmup finding (`2142b55`, `RESULT.md` §2) is **EXPLORATORY**: warmup's scores were already known and the clause-fired split is defined by the scorer's own output, so it is a post-hoc split on a seen venue. It has **16 clusters**.

navhard has **450 clusters** — ~28× — and **no navhard score has been seen by the author of this document.** The EvalFlyWheel's run (`taniteval.bench navsim_v2 --ckpt none --split navhard_two_stage --arms CV,STOP`, started 10:28) is still in flight; nothing from it has been read.

⭐ **This is the second time today the same window has been open, and the first time it produced a refutation of my own derivation.** Writing it now is what makes the replication confirmatory rather than a story told after the fact.

## 1 · The directional prediction, committed

From warmup: STOP − CV is **+0.1114**, CI [0.0292, 0.1821], on the scenes where the clause did **not** fire.

**Primary endpoint:** `STOP − CV` on navhard's **clause-NOT-fired** stratum, paired, bootstrapped over **`orig_scene` clusters** (B = 10,000).

| outcome | reading |
|---|---|
| **> 0, CI excluding 0** | ⇒ **CONFIRMED at 450 clusters.** The ≤ 5 m clause is an amplifier and **not** the cause of STOP's win; `D-NAVSIM-STOP-1`'s second mechanism owns the effect |
| **CI includes 0** | ⇒ **NOT replicated.** The warmup result was a 16-cluster artifact. ⛔ Report it as a failed replication; do not rescue it by re-slicing |
| **< 0, CI excluding 0** | ⇒ **REVERSED.** STOP loses once the clause is removed on the larger venue, and the clause *was* the cause — the warmup reading was wrong in sign |

**Secondary, reported always, never instead:** the **sign test** (scenes STOP wins / CV wins / ties). ⛔ On warmup these **disagreed** with the mean — CV won 83 scenes to STOP's 68 while the mean favoured STOP. Both are reported or neither is.

**Also reported:** `f_fired` on navhard, and the FIRED-stratum gap, so the amplifier's size is measured rather than assumed.

## 2 · ⛔ Fixed before the numbers

* **Clause detection:** `ego_progress_stage_two == 1.0` for **every** arm scored. ⚠️ With only CV and STOP available this is a **weaker** detector than warmup's six-arm version — two arms agreeing at 1.0 is a lower bar than six. If additional arms are scored, use all of them, and report **how many arms** the detector used beside the number.
* **Cluster key:** `corresponding_original_scene` from `raw/navhard_token_v0.csv` (450 clusters, median 12 scenes each). ⛔ No interval over scenes.
* **Bootstrap:** paired, B = 10,000, seed **20260920**, resampling clusters with replacement.
* ⛔ **No threshold, no speed stratification, no subgroup not named here.** The 5.0 m/s stratum is refuted (`RETR-2026-09-20-FASTSTART-THRESHOLD`) and must not reappear.

## 3 · What this cannot settle

⚠️ **It tests the clause's role, not the remaining mechanism.** A zero-displacement plan earns EP median **0.195** on warmup's non-fired scenes (`d86dccb`) and *why* is still unknown. Every outcome above leaves that open, and none may be quoted as bearing on it.

⚠️ **STOP and CV are deterministic**, so training and inference variance are zero by construction and the cluster bootstrap is the whole question here — **but that property belongs to these two arms only** and does not transfer to any trained arm scored on this stratum later.

⚠️ **navhard and warmup are different venues.** A replication across them is stronger than a re-analysis within one and weaker than a pre-registered split of a single venue. Say which, in the result.

<!-- INSTRUMENT-AND-PRECISION-2026-09-20 -->

## 4 · The analysis instrument exists BEFORE the data, and it is self-tested

`code/confirm.py` implements §1–§2 exactly. It was written while only `CV.csv` existed and
`STOP.csv` did not, so the navhard run is a single invocation with nothing left to choose.
⭐ Every analyst degree of freedom that survives until after the numbers arrive is a place a
result can be steered.

`--self-test` runs the identical code path on **warmup**, whose answers are landed at `2142b55`:

| check | basis |
|---|---|
| `n_scenes` 204 · `n_fired` 37 · clusters 16 · gap **0.1114** · gap-fired **0.2179** · wins **68/83/16** | **exact** — deterministic |
| CI bounds | **within measured Monte-Carlo error** — resampled quantiles are not deterministic |
| **VERDICT** | **exact** — it is what the prereg acts on |

⛔ **And it carries DELIBERATE-REGRESSION arms**, because a guard that only ever passes is not
evidence. The self-test re-introduces the two defects that actually happened — joining on
`scene_token`, and a one-arm clause detector — and FAILS if either is tolerated. Both are
rejected (`ZZABORT ... empty join`; `ZZREFUSE ... 1 arm(s)`).

## 5 · ⚠️ A PRECISION CORRECTION to the interval this replication is measured against

The self-test reproduced every point estimate exactly and disagreed with the banked CI in the
**3rd decimal** — a different RNG stream, not a defect. Measured directly (`code/mc_error.py`,
**12 independent streams**, B = 10,000, on the 16-cluster fixture):

| bound | mean | sd | range over 12 streams | spread |
|---|---|---|---|---|
| lower | 0.0309 | **0.00166** | [0.0281, 0.0333] | 0.0052 |
| upper | 0.1819 | 0.00066 | [0.1806, 0.1831] | 0.0025 |

⇒ **`2142b55` reported `[0.0292, 0.1821]` to 4 dp; the procedure supports about 3.** Intervals
are now printed at **3 dp** with their Monte-Carlo error stated.

⭐ **What IS stable is the verdict: the lower bound exceeds zero in 12/12 streams** (min 0.0281).
So the finding is unchanged and only its reported digits were over-precise — say both.

⛔ **The seed was NOT tuned until the bounds matched.** That would have fitted the instrument to
its own fixture, which is the *"a check that shares the defect it checks for"* failure in a new
costume. The tolerance was set from the measured spread instead.
