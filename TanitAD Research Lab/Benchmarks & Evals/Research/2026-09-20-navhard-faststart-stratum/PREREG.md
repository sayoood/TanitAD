<title>navhard fast-start stratum prereg</title>

# `D-NAVSIM-STRATIFY-1` — the fast-start threshold is **5.0 m/s**, fixed here, derived from the scoring rule, before any navhard score exists

`Benchmarks & Evals · 2026-09-20 · Master Mind · PRE-REGISTRATION — MEASURED join key, 0 GPU, no score seen`
`Instrument: code/tokens.py → raw/navhard_token_v0.csv (the join key) + raw/navhard_stratum.json. Threshold constants re-read from the installed devkit at C:/Users/Admin/navsim-crun/devkit.`

## 0 · The ordering, which is the only thing that makes this a pre-registration

⛔ **No navhard score has been seen by the author of this document.** The one navhard run in flight is the EvalFlyWheel's `taniteval.bench navsim_v2 --ckpt none --split navhard_two_stage --arms CV,STOP`, started 10:28 and still scoring STOP when this was written. Nothing from it has been read.

⭐ **And the threshold does not come from a distribution at all** — not from the scores (unseen) and not from the start speeds either. It is derived from the scoring rule, below. The speed distribution enters exactly once, *after* the number was fixed, to report the resulting **n**. That ordering is the difference between a stratum and a subgroup found by looking.

⚠️ This is written now **because the window closes.** Once Eval's CV/STOP numbers land, no threshold chosen afterwards can be shown to be score-blind, however honestly it was picked.

## 1 · The threshold, and why it is this number

The clause the hypothesis is about, read from the installed devkit rather than carried forward:

```python
# navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:232-237
masked_progress = self._progress_raw * multiplicate_metric_scores
norm_constant_progress = np.max(masked_progress)
if norm_constant_progress > self._config.progress_distance_threshold:
    normalized_progress = np.clip(self._progress_raw / norm_constant_progress, 0.0, 1.0)
else:
    normalized_progress = np.ones(len(masked_progress), dtype=np.float64)   # <- EP = 1 for EVERY proposal
```

| constant | value | read from |
|---|---|---|
| `progress_distance_threshold` | **5.0 m** | `config/pdm_scoring/scorer/pdm_scorer.yaml:20` (= the dataclass default, `pdm_scorer.py:58`) |
| agent `time_horizon` | **4 s** | `config/common/agent/{constant_velocity,ego_status_mlp,human,transfuser}_agent.yaml:7` — all four agree |

⇒ **A vehicle that merely MAINTAINS 5.0 m/s covers 20 m in 4 s — 4× the clause distance.** So on a scene starting at or above 5.0 m/s, the clause cannot fire *on progress grounds* against any plan that roughly holds speed. **`THRESHOLD = 5.0 m/s`, fixed.**

⚠️ **What the derivation does NOT claim.** The clause tests the best **masked** progress — raw progress zeroed wherever a multiplicative metric (collision, drivable area) fails. A fast start therefore does not *guarantee* the clause stays silent: a scene that starts at 8 m/s into a blocked junction can still afford ≤ 5 m of rule-compliant progress. That residue is not a defect in the threshold, it is the thing the control in §3 measures.

## 2 · The resulting n, reported after the number was fixed

| stratum | criterion | n | share |
|---|---|---|---|
| **FAST** | \|v0\| ≥ 5.0 m/s | **2,116** | **38.7 %** |
| SLOW | \|v0\| < 5.0 m/s | 3,346 | 61.3 % |
| total | all navhard synthetic scenes | **5,462** | 100 % |

**Join key: `raw/navhard_token_v0.csv`**, 5,462 rows of `scene_token,v0_ms`, one per synthetic scene, tokens asserted unique. This is the table that did not exist before — the banked start-speed run (`6a1d732`) kept the speeds but **not** the tokens, so no stratum could be joined to any per-scene dump. It exists now.

⭐ **The stream was re-run from scratch and reproduced the banked distribution exactly** (median **3.89 m/s**, n **5,462**, 0 unreadable) — an independent replication of `6a1d732`'s numbers, not a re-read of them.

## 3 · The endpoints, committed before any score is read

**Primary endpoint:** the **≤ 5 m clause's firing fraction on the FAST stratum**, `f_fast` — the share of scored fast-start scenes where `norm_constant_progress ≤ 5.0`, so EP is forced to 1 for every proposal. On warmup, pooled, it is **37/204 = 18.1 %** (`d86dccb`, INHERITED from our own landed census).

| outcome | reading |
|---|---|
| **`f_fast` < 5 %** *and* STOP's EPDMS advantage over CV on FAST has a paired-bootstrap CI that **includes or sits below 0** | ⇒ the clause **is** the mechanism: remove the venue where it fires and stopping stops winning |
| **`f_fast` < 5 %** *and* STOP still beats CV with a CI **excluding 0** | ⇒ ⛔ the clause is **NOT** what makes stopping win. The threshold did its job and the hypothesis is refuted anyway — this is the informative failure |
| **`f_fast` ≥ 15 %** | ⇒ ⛔ **the derivation in §1 is wrong** — a fast start does not buy 20 m of compliant progress in this corpus. Report it as a failed precondition and do **not** reinterpret the arm comparison |

⛔ **Arm-independent control, and it is what makes the stratum load-bearing rather than decorative:** `f_slow` must exceed `f_fast`. If the clause fires at the same rate in both strata, start speed is not what drives it and the split separates nothing — in that case the arms are uninterpretable regardless of which way they come out. **This control is computed from the scorer's own per-scene output and needs no arm**, exactly as the 18.1 % census did.

⚠️ **And one number is already known to be immune to all of this:** the clause explains 18.1 % of warmup scenes; the remaining **82 %** is **UNEXPLAINED**, with a zero-displacement plan still earning EP median **0.195** there (`d86dccb`). **This prereg tests the clause only.** `D-NAVSIM-STOP-1`'s second mechanism is untouched by every outcome in the table above, and no result here may be quoted as bearing on it.

## 4 · Scope, and what may not be inferred

* This stratifies **navhard's synthetic scenes**. Which of them a given bench run actually scores is that run's business; the join is by `scene_token`, so a run that scores a subset joins cleanly and reports its own n.
* ⛔ **The venue ruling stands and is not reopened by this.** `H-NAVHARD-STOP-1` lost navhard-as-a-whole as a venue because navhard's starts are not faster than warmup's (median 3.89 vs 4.14 m/s, `6a1d732`). This does not restore that comparison — it replaces it with a **within-corpus** contrast, which is a different and weaker design: FAST and SLOW here differ in start speed *and* in whatever else co-varies with start speed (junctions, traffic density, scene type). It is not a randomised split and must never be written up as one.
* ⚠️ The confound above is the honest limit of this design. The control in §3 constrains it but does not remove it.

## 5 · ⚠️ A citation correction, recorded because the programme rule is that every number cites a file

The clause has been cited as **`pdm_scorer.py:231-236`** in four landed documents and as **`232-237`** in a fifth — so the repository contradicted itself, and a reader could not tell which was right.

**`232-237` is correct.** Line 231 is a comment (`# normalize and fill progress values`) and 236 is the bare `else:`; the range `231-236` therefore **excluded line 237** — `normalized_progress = np.ones(...)`, the assignment the entire claim is about. A reader following the citation would have landed on the branch test and not on its effect.

⭐ The defect is small and the lesson is not: the span was copied forward five times without anyone opening the file. Both constants in §1 were re-read from the installed devkit for this document rather than carried from a summary — which is how the off-by-one surfaced at all.

<!-- AMENDMENT-1-THRESHOLD-REFUTED-BY-ITS-OWN-CONTROL-2026-09-20 -->

---

## 6. ⛔ AMENDMENT 1 — the 5.0 m/s threshold is REFUTED by the control this document committed to

Measured the same day, before any arm ran, with no new scoring: on warmup the clause fires on
**32.5 %** of fast starts (27/83) and **8.3 %** of slow ones (10/121). §3 committed *"`f_fast`
≥ 15 % ⇒ the derivation is wrong"*; it reads **32.5 %**, and `f_slow` is **0.25×** `f_fast`
rather than greater, so the arm-independent control fails **by sign**.

⭐ **The error is nameable in one sentence:** §1 priced how far a vehicle **travels**; the clause
prices how far a **rule-compliant** proposal travels. At fast starts a moving arm's multiplicative
compliance collapses (CV: no-at-fault-collision **0.386 vs 0.884**, drivable-area **0.566 vs
0.818**, driving-direction **0.596 vs 0.959**), so every compliant proposal is a slow one and the
best masked progress stays under 5 m **because** the scene is fast.

⚠️ §1 already carried this exact caveat and called it a *"residue"* the control would measure.
It is the dominant term. The caveat was right and its weighting was wrong — which is the reason
the control was committed in advance rather than argued about afterwards.

⛔ **Consequently §3's endpoints are VOID and the five arms must not be run on this stratum.**
⛔ **And §2's and §4's `n` are wrong in UNIT:** navhard's 5,462 synthetic scenes derive from
**450 original scenes** and warmup's 204 from **16** (`corresponding_original_scene`), so any
interval over scenes is pseudo-replication.
⛔ **And §2's join key names the wrong field:** `scene_token` joins **0** of 204 against the
scorer's per-scene CSVs; **`initial_token`** joins 204/204. `raw/navhard_token_v0.csv` is
re-emitted as `initial_token,scene_token,orig_scene,v0_ms`.

⇒ What survives is the **method**, not the threshold: fix the stratum from the rule, commit the
failure criterion, and run an arm-independent control **before** the arms. Here that machinery
refuted its own author's derivation for the price of one join. See `RESULT.md` §2 for the result
it produced anyway — the clause is **not** what makes stopping win.
