<title>Clause confirmed as amplifier</title>

# `D-NAVSIM-CLAUSE-STRATUM-1` — **CONFIRMED at 435 clusters**: the ≤ 5 m EP clause is an amplifier, **not** the cause of STOP's win. And the eligibility-bias worry is refuted by its own interval.

`Benchmarks & Evals · 2026-09-20 · Master Mind · MEASURED, 0 GPU — the pre-registered analysis (b66e8cd) run unchanged on navhard`
`Instrument: code/confirm.py (self-tested + 2 deliberate-regression arms) on the EvalFlyWheel's navhard CV/STOP run → raw/confirm_navhard.json`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⭐⭐ **CONFIRMED.** On the **4,240 scenes / 435 clusters** where the clause does **not** fire, `STOP − CV` = **+0.113**, CI95 **[0.099, 0.127]**, separated. Warmup gave **+0.1114** on 16 clusters. **Two venues, 27× the clusters, agreement to three decimals.** | MEASURED |
| **2** | The clause **amplifies ~2.1×** where it fires: **+0.2376** [0.214, 0.261] on 1,222 scenes / 345 clusters, against +0.2179 on warmup. ⇒ it is a real effect and it is **not** the cause. | MEASURED |
| **3** | ⚠️ **The mean/sign divergence did NOT replicate.** On warmup's not-fired scenes CV won *more often* (83 vs 68) while the mean favoured STOP. On navhard both agree: **STOP 1,913 / CV 1,655 / ties 672**. The warmup divergence was a 16-cluster artifact. | MEASURED |
| **4** | ⛔ **The eligibility-bias worry is REFUTED**, and by its own interval. The dominant refusal `future` (22.81 %) is **not separated** from KEPT on near-forward target count or range. | MEASURED |
| **5** | ⚠️ **A defect in my own instrument, found in its output:** `_CI_mc_error` is a hardcoded literal reading *"B=10,000 on 16 clusters"* — true of the warmup fixture, **false** on navhard's 435 clusters, where the Monte-Carlo error is far smaller. Corrected. | MEASURED |

## 1 · The pre-registered endpoint, run unchanged

⛔ The prereg (`b66e8cd`) was written while only `CV.csv` existed; `STOP.csv` did not. It fixed the endpoint, the estimator, the seed, the cluster key and **the outcome that would refute the warmup reading**. The analysis below is that file executed, with nothing chosen afterwards.

| stratum | scenes | **clusters** | `STOP − CV` | CI95 (orig_scene bootstrap) | STOP / CV / ties |
|---|---|---|---|---|---|
| clause **NOT fired** ⭐ *primary* | 4,240 | **435** | **+0.113** | **[0.099, 0.127]** | 1,913 / 1,655 / 672 |
| clause **FIRED** | 1,222 | 345 | **+0.2376** | [0.214, 0.261] | 468 / **34** / 720 |
| all | 5,462 | 450 | +0.1409 | [0.129, 0.153] | 2,381 / 1,689 / 1,392 |

`f_fired` = **0.2237** on navhard against **0.1814** on warmup.

⇒ **Remove every scene where the clause fires and stopping still wins, separated, on 435 independent clusters.** `D-NAVSIM-STOP-1`'s **second mechanism** — a zero-displacement plan earning EP median 0.195 where the clause is silent (`d86dccb`) — now owns the effect and is the programme's open question here.

⚠️ **Pre-stated and still true: the detector is weaker on this venue.** It is `ego_progress_stage_two == 1.0` across **every arm scored**, and this run has **2 arms** (CV, STOP) against warmup's **6**. Two arms agreeing at 1.0 is a lower bar, so `f_fired` = 22.4 % is an **upper** estimate of the clause's true firing rate and the 4.2-point gap over warmup is consistent with mild over-counting. ⛔ This cuts *against* the finding's convenience, not for it: over-counting fired scenes moves genuinely-not-fired scenes **into** the fired stratum, which would if anything deflate the not-fired gap.

## 2 · ⛔ The eligibility-bias worry, refuted by the interval I insisted on

An indicative first pass (n = 148 vs 112, **no interval**) suggested refused windows carried more and closer near-forward targets — which would have meant the perception bar was measured on a subset under-representing dense close traffic. Decomposed by reason, episode-clustered:

| group | windows | episodes | near-forward targets | CI95 | mean range (m) | CI95 |
|---|---|---|---|---|---|---|
| **KEPT** | 251 | 58 | 5.096 | [3.761, 6.537] | 52.32 | [45.833, 59.321] |
| `future` (22.81 %) | 246 | 59 | 5.207 | [3.729, 6.809] | 53.09 | [45.334, 62.183] |
| `agents` (5.97 %) | 33 | **4** | **0.091** | [0.0, 0.75] | 91.65 | [38.798, 114.636] |

⇒ **The dominant refusal is unbiased.** `future` overlaps KEPT on both measures. ⭐ And `agents` separates in the **opposite** direction to the worry: it removes windows with essentially **no** near-forward targets — near-empty scenes, not dense ones. ⚠️ On only **4 episodes / 33 windows**, so weak; it is enough to refute the stated worry, not to support a claim of its own.

⭐ **The lesson is about my own first pass.** KEPT's near-forward count read 4.02 there and 5.096 here — the *same statistic* on a different draw — while the CI spans **2.8**. The apparent 4.02-vs-5.11 gap was inside the noise the whole time. **I reported it as indicative and refused to assert it; adding the interval is what killed it.** Same family as `CLASS-2026-09-20-REPLICATION-UNIT` from the other side: there an interval was too narrow for its unit, here a difference was quoted before it had one at all.

## 3 · ⚠️ The defect in my own instrument

`confirm.py` emits `_CI_mc_error` as a **hardcoded string**: *"B=10,000 on 16 clusters: lo sd 0.0017, hi sd 0.0007 — 3 dp"*. That was measured on the warmup fixture and it is **wrong on navhard**, where 435 clusters make the Monte-Carlo error much smaller. It sat in all three strata of the output.

⛔ A provenance note that does not travel with its data is the same failure the programme keeps logging — **a true quantity quoted outside its scope**, here inside the very instrument built to stop it, one turn after I built it. Fixed to carry the *measured* cluster count, or nothing.

⚠️ **No number moves.** The literal is annotation, not input; every interval above was computed from navhard's own resampling.

## 4 · What this does and does not settle

* ⛔ **`H-NAVHARD-STOP-1`'s mechanism question is answered: not the clause.** The venue ruling stands separately — navhard cannot test the *original* hypothesis (starts are not faster: 3.89 vs 4.14 m/s) and this is a different, stronger design.
* ⛔ **It says nothing about why a stationary plan scores at all.** EP median 0.195 on non-fired warmup scenes remains **unexplained**, and no outcome here bears on it.
* ⚠️ **STOP and CV are deterministic**, so training and inference variance are zero by construction and the cluster bootstrap is the whole question — **for these two arms only.** No trained arm inherits that property.
