# PROGRAM HARVEST — systematic sweep of all 135 agent deliverable directories

**Date:** 2026-07-26 · **Commissioned by Sayed:** *"look on all our program agents how we can leverage
their results and incorporate them in the recovery plan."* · **Feeds:** `Project Steering/BOOST_PROGRAM.md`
**Compute:** CPU only, no pod, no GPU (pod1 training · pod2 H2 classifier · eval pod Bar-A).

## TL;DR — six sentences

1. **789 of our "no difference" findings sit at n≈40 episodes, and 516 of them (65 %) would separate at
   n=600 on the measured ×3.4 half-width shrinkage** — we do not know what most of our nulls say.
   *(MEASURED; the count is CONFIRMED by an independent grep; the projection is arithmetic, PROVISIONAL.)*
2. **Half of that is dangerous, not lost value:** for the **firewall / leakage / shuffle controls** a null
   is the *desired* verdict, so "not separated at n=40" is **a leak we could not see** — re-adjudicate
   those first, because they can only ever REMOVE results.
3. **The clearest proof it is a power problem, not an effect problem:** two artifacts measure the *same*
   0.20 m closed-loop effect on the *same* 12 episodes with the *same* estimator — one is published as
   **"the proof"**, the other as **"TIE — do not promote"**. The second misses zero by **0.0154 m**.
4. **Three things are live and will cost something today:** a standing directive commissions a ~1-pod-day
   probe that is **already complete at five locations** (O-1) · the module built to un-strand the v4
   co-primary **raises on its first step**, one line (H3 S-01) · and **`CLAUDE.md` — loaded into every
   agent's context — cites a claim its own `RETRACTION_LOG` retracted on 07-21** (O-5).
5. **The recovery plan's Bar-B pessimism rests on a trainer-log number** (−21.6 %/20k) while the
   eval-grade pair reads −45.0 %/15k from a *different statistic* — and **Bar B's first candidate lever
   now has two independent supports** (off-path/viewpoint augmentation, H5 row 4).
6. **`BOOST_PROGRAM` §4 still lists three streams against the PI's "at least five"** — this harvest
   supplies the missing two, **both eval-only**, so breadth costs no GPU contention.

⛔ **Nothing in this document is DECISION-GRADE.** It is a ranked list of places to look, built from
reports, and reports are `INHERITED` by default. Where I verified something myself with two probes it is
marked `MEASURED · CONFIRMED`; everywhere else it is `PROVISIONAL` and must not decide a GPU-day.


## Method, and its limits — read before quoting anything below

| what | how |
|---|---|
| corpus swept | **135** dirs under `TanitAD Research Hub/*/Implementation/incoming/*/` + `*/Research/` + `Project Steering/` + `Reviews/` + `taniteval/results/` — ⚠️ **the count is moving under us and nobody's number is wrong**: `BOOST_PROGRAM.md` says **134**, I counted **135** at sweep start, the H3 agent counted **136** mid-sweep, and a recount at write-up reads **137** (this harvest dir + `…/2026-07-26-publishable-corpus-hunt/`, which appeared at 19:39 today). **That is itself a datum about the harvest problem** — the corpus grows faster than any single read of it. |
| files | **642 JSON parsed** · 222 `.md` (3.19 MB) grepped · 373 `.py` |
| H1 instrument | `scratchpad/h1_sweep.py` → `h1_rank.py` → `h1_project.py` (recursive walk, every node carrying `separated: false`, sibling effect + CI + n extracted, `\|effect\|/half-width` computed). **Staged as `artifacts/h1_sweep.py` etc.** so this is re-runnable and diffable. |
| **evidence class of this document** | The *sweep statistics* (counts, proximities, projections) are **MEASURED (ours, `harvest_index.json`)** — I computed them from the raw JSON, not from prose. The *claims the artifacts make* are **INHERITED** and marked so per row. |
| **tier** | The sweep is **PROVISIONAL** — one path, one agent. **Exception, and it is the load-bearing count:** the headline `separated: false` total was re-derived by an **independent tool** (`grep -rc '"separated": false'`) giving **2050** in `TanitAD Research Hub` + `taniteval` and **1** in `Project Steering` + `Reviews` = **2051**, matching the parser exactly ⇒ **that count is CONFIRMED.** Everything derived from it (bucketing, proximity, projection) remains PROVISIONAL. **Nothing here is DECISION-GRADE.** |
| ⚠️ estimator hygiene | Nodes produced by `overlapping_holdout_se` / `_jack` are **excluded from the results tables and listed separately as needing re-estimation** (it biases point estimates −6.67 %…+11.69 %, up to ×−4.15 with sign flips). |
| 🔒 | No clip UUIDs or PhysicalAI raw content appear in any artifact here. |

**Known limits, stated rather than hidden:**
1. The sweep sees `separated: false` **as emitted in JSON**. Verdicts written only in prose ("we see no
   difference") with no JSON node are **NOT** in the H1 counts. H1's md-grep found 87 such files; the
   ones I read are folded in by hand and marked `prose-only`.
2. `proximity = |effect| / half-width` is a **screening** statistic, not a power calculation. It ranks
   *closeness to separation*; it does **not** predict a flip, because the point estimate can move.
3. I could not reach any pod. **Checkpoint existence is INHERITED from `MODEL_REGISTRY.md`**, not probed.

---

# H1 — n≈40 nulls that must be re-adjudicated at n=600 ⭐ MAIN DELIVERABLE

## H1.0 The headline number

`MODEL_REGISTRY.md` §1.2a establishes MEASURED that the 600-episode val is an **order-preserving
superset** of the canonical 40, that CI half-widths shrink **×2.8–3.9 (mean 3.4)**, and that one verdict
already flipped on power alone (`along_track_vs_cv`: `[−0.0278, +0.5304]` "tie" → `[+0.1926, +0.3104]`
"model wins", point estimate moving **0.7 %**). Its consequence line: *"any verdict here that rests on a
40-episode 'not separated' is **UNPOWERED, not refuted**."*

**Applied to the whole program, MEASURED by this sweep:**

| quantity | value | class / tier |
|---|---:|---|
| JSON files parsed | 642 | MEASURED · PROVISIONAL |
| `separated: false` nodes found | **2 051** | MEASURED · PROVISIONAL |
| …with an extractable effect + interval | 1 785 | MEASURED · PROVISIONAL |
| …after de-duplicating the `X.json` / `driving_X.json` twin dumps | **1 502** | MEASURED · PROVISIONAL |
| …computed with a **valid** estimator (paired/episode-cluster bootstrap) | **1 480** | MEASURED · PROVISIONAL |
| …computed with the **deprecated `_jack`** ⇒ not results, need re-estimation | **21** | MEASURED · PROVISIONAL |
| **nulls at n≈36–43 episode clusters (the 600-ep target)** | **789** | MEASURED · PROVISIONAL |
| **…that would separate at n=600 under the mean measured ×3.4 shrinkage** | **516 (65.4 %)** | MEASURED (arithmetic) · PROVISIONAL |
| …that would separate even under the **weakest** measured ×2.8 | **457 (57.9 %)** | MEASURED (arithmetic) · PROVISIONAL |
| nulls at n≈12–24 (the closed-loop suites) | **466** | MEASURED · PROVISIONAL |

⚠️ **How to read "would separate".** This is *arithmetic on the interval only*: it asks whether
`|effect| / (half-width ÷ 3.4) > 1`. It assumes the point estimate holds. In the **one** case we have
observed end-to-end it did hold (0.7 % movement), but **n=1**, and this document does not claim more
than that — this is a **ranked hypothesis list, not a list of results**. The correct statement is:
*we do not know what 516 of our 789 "no difference" findings actually say.*

**Calibration check that makes the ranking credible:** the one verdict already known to have flipped —
`vs_floor_paired.cv.long_abs_2s_m` on `flagship-30k` — scores **proximity 0.911** in this sweep, i.e. it
sits in the top ~5 % of the n≈40 population. The instrument ranks the known positive where it should.

## H1.1 ⭐ RANKED — the workstream-level verdicts that closed a direction

One row per workstream, its **closest-to-separation non-panel null**. Ranked by proximity.
`prox@600` = proximity × 3.4 (mean measured shrinkage). All effects/intervals **MEASURED by the owning
agent**; my re-extraction of them is MEASURED; **the interpretation column is INHERITED · PROVISIONAL.**

| # | workstream / what was concluded | the null (paired Δ, CI95) | n eps | **prox** | **prox@600** | ckpt still exists? | what a re-score could recover |
|---|---|---|---:|---:|---:|---|---|
| **1** | **`2026-07-23-freefloor-rung3-wm-mpc`** — verdict **"TIE — WM-MPPI/CEM planning does NOT beat single-step re-plan; do NOT promote rung 3"** | `headline.paired_delta_C_minus_A_ade@2s` = **−0.2016 [−0.3616, +0.0154]** | **12** | **1.069** | **3.64** | ✅ v1 = `flagship-30k`, **3 copies** (HF gated + eval pod + pod2) | The *imagination* leg (C vs single-shot A) is a **0.20 m effect one hair from separation at n=12**. The TIE that was published is C-vs-B; C-vs-A was never the headline and is nearly significant. Re-score at n=600 (or even n=40) is **hours of eval, zero training**. |
| **2** | **`2026-07-26-e1c-heldout-gated-clsft`** — E1c frontier, held-out-gated CL-SFT | `points.2500…frenet.paired_delta_ft_minus_base.along_abs@2s` = **+0.1338 [−0.0002, +0.2585]** | 43 | **1.034** | **3.52** | ⚠️ E1c frontier ckpts — **verify**, produced this week | Lower bound is **−0.0002**. This is separated to four decimal places away. The E1c frontier's along-track cost may be real and was read as "no cost". |
| **3** | **`2026-07-23-v4-eval-harness`** — v4.2 @4000 vs CV floor | `cluster_bootstrap.model_vs_cv_paired` = **−0.1492 [−0.2980, +0.0256]** | 40 | 0.922 | **3.14** | ✅ `tanitad-eval:/root/models/flagship-v4.2-step4000/ckpt.pt` (md5 in registry §1.5) | "v4.2@4k does not beat CV" is one of the readings that fed the v4 line's pessimism. At n=600 this is a **3.1σ-equivalent** margin. |
| **4** | **`2026-07-21-lead-state-gate`** — the gate that **REFUSED a 12.4 GB + 2–3 eng-day `obstacle.offline` ingest** | `ridge\|canonical.paired_mae_A_minus_B_shuf` = **−0.0014 [−0.0029, +0.0002]** | **126** | 0.903 | 3.07 | n/a — this is a **ridge/GBM probe on features, no model ckpt needed** | ⚠️ **Read the row carefully: the near-separated node is the SHUFFLE control, not the treatment.** The treatment (`A_minus_B` = +0.0051 [−0.0040,+0.0143], prox 0.557) is *further* from separation and is the **wrong sign** (lead state made it worse). **The refusal stands; more power does not rescue it.** Recorded here to close it, not to reopen it. |
| **5** | **`2026-07-26-v4-produced-goal`** — goal-mode gap, lateral/longitudinal decomposition | `cross.paired_produced_minus_oracle_meanhorizon` = **+0.0114 [−0.0012, +0.0242]** | 40 | 0.898 | **3.05** | ✅ v4 fromscratch 15k/30k on pod2+eval | "the oracle privilege is not cross-track" — at n=600 it probably **is**, by ~11 mm. Small but it changes the mechanism story for Bar A. |
| **6** | **`2026-07-23-refc-planner-closedloop`** (gentle sweep G1) | `paired_base_minus_ft.overall.delta_peak_xte_m_base_minus_ft` = **−0.2474 [−0.5280, +0.0262]** | 12 | 0.893 | **3.04** | ✅ REF-C base/XL, **3 copies each** (HF gated + eval pod + pod3) | **0.25 m of peak cross-track error** read as "no effect" on 12 episodes. This is the road-keeping axis the whole closed-loop program is stuck on. |
| **7** | **`2026-07-24-refccl`** — tolerance re-score, the artifact that reopened *then* closed the CL direction | `arms.naive.bands.0.5.delta_corridor_departure_base_minus_ft` = **−0.0379 [−0.0833, +0.0032]** | 12 | 0.876 | **2.98** | ✅ same REF-C ckpts | The report's own conclusion is *"CI∋0 for 3/4 configs"*. **At n=600 the 0.5 m band deltas plausibly all separate.** 33 non-panel nulls in this one workstream; the top six all project > 2.6. |
| **8** | **`2026-07-22-imagination-closedloop-proof`** ⇄ **`2026-07-26-closedloop-artifact-rerun`** | `imagination_comparison.paired_delta_B_minus_A_fde@2s` = **−0.4176 [−0.8570, +0.1118]** | 12 | 0.862 | **2.93** | ✅ v1, 3 copies | The **ADE** leg separated and became "the proof"; the **FDE** leg did not and was dropped. FDE is the endpoint error — the quantity closed-loop safety actually cares about. Recovering it would **strengthen** the imagination thesis, which is v4's whole premise. |
| **9** | **`2026-07-25-closedloop-horizon-and-shift`** (E1a) | `deltas_vs_K20.other.40.d_window_departure_rate` = **+0.0417 [−0.0049, +0.0885]** | 20 | 0.862 | **2.93** | ✅ v1 + v4 | E1a's *junction* stratum separated hard and became the C9 class; the **"other" (non-junction) stratum** was read as flat. It probably is not. That changes "junction-specific" to "general". |
| **10** | **`2026-07-26-4brain-s3`** / **`2026-07-26-s3-decision-grade`** — blind-baseline firewall | `lon.paired_leak_B2_minus_B1` = **+0.1657 [−0.0451, +0.3518]** | 73 | 0.835 | **2.84** | ✅ | ⚠️ **INVERTED VALUE — see H1.3.** This is a **leak/firewall control**, where a null is the *desired* result. Projecting it to n=600 says the S3 firewall may be **hiding a real leak of 0.166**. This is the single most dangerous row in the table. |
| **11** | **`2026-07-26-wheelbase-impact`** — "the wrong wheelbase costs little" | `paired_vs_shipped.corrected.cross_cum_0_2s` = **+0.0088 [−0.0005, +0.0207]** | 40 | 0.830 | **2.82** | ✅ v1 | 18 non-panel nulls; top four all project > 2.5. The wheelbase decision (PI decision item, §5.6 of BOOST) was taken partly on *these* nulls. **Cheapest re-adjudication in the table** — it is a data-side re-score, no training. |
| **12** | **`2026-07-25-e1b-failure-gated-clsft`** — the pre-registered **BOUND** verdict | `closed_loop_K20.longitudinal.win_dep.paired_delta_ft_minus_base` = **+0.0241 [+0.0000, +0.0581]** | 27 | 0.830 | 2.82 | ⚠️ E1b FT ckpt — **verify it survived** | Lower bound is **exactly +0.0000**. The BOUND verdict stands on the *guardrails* (which failed hard and separately), so this does **not** reopen it — but the *effect* leg was mis-read as absent. |
| **13** | **`2026-07-23-dagger-closedloop-aware`** — verdict **"DAGGER_HURTS → do not promote"** | `A1_bc_minus_A0_baseline.open_loop_head_ade2s` = **+1.4398 [−0.1543, +3.3888]** | 12 | 0.813 | **2.76** | ✅ v1 | The *BC control* leg. If this separates, the "matched-budget BC control" that made the DAgger verdict clean was **itself** moving the arm — i.e. the DAgger comparison was **confounded by its own control**. High value, cheap. |
| **14** | **`2026-07-26-v4-30k-gate`** (co-primary) | `paired_v4_vs_refc_K20.junction.d_corridor_departure_rate` = **−0.0269 [−0.0632, +0.0040]** | 22 | 0.801 | 2.72 | ✅ v4 30k + REF-C | "v4 ties REF-C on junction departure at K=20" — the co-primary the RESTART verdict rests beside. |
| **15** | **`2026-07-26-v4-restart-lever`** | `paired_as_trained_minus_zero.miss_at_2m` = **−0.0068 [−0.0154, +0.0017]** | 40 | 0.800 | 2.72 | ✅ | The `sel_gate := 0` counterfactual separated on ADE (the retraction-logged good catch) but **not** on `miss_at_2m`. At n=600 it likely does — strengthening the "the gate helps" finding. |
| **16** | **`2026-07-26-h2-classifier`** | `operating_point.paired_recall_deltas.head_ego - random_at_rate` = **+0.1340 [−0.0339, +0.3019]** | 322 | 0.798 | 2.71 | ✅ pod2 (**training now** — do not touch) | H2's classifier "does not beat random at rate". n=322 already; the 600 target does not apply, but a **larger clip draw would**. |
| **17** | **`2026-07-21-vlm-production-semantic`** | `paired_tests.tests[2].paired_ci` = **+0.0536 [−0.0329, +0.1401]** | 14 | 0.620 | 2.11 | n/a — VLM probe, re-runnable on CPU/eval | VLM semantic labelling was scoped down on nulls at **n=14**. |
| **18** | **`2026-07-20-vlm-reason1-vs-reason2`** | `paired_tests.tests[0].paired_ci` = **−0.0409 [−0.1142, +0.0324]** | 38 | 0.558 | 1.90 | n/a | "Reason1 and Reason2 are indistinguishable" — a **model-choice** decision taken on an n=38 null. |
| **19** | **`2026-07-25-jack-blast-radius`** | `paired[1..2].paired_bootstrap` = **+0.0443 / +0.0457, hw ≈0.10** | 40 | 0.441/0.443 | 1.50/1.51 | ✅ | This is **REF-C-XL vs flagship v1** — the registry's *"they tie"*. **At n=600 this plausibly separates, i.e. the program has a rank order between its two best arms that it currently believes it does not have.** |
| **20** | **`2026-07-25-v16-paired-interval`** | `SECONDARY_v16_vs_refc_xl_G1` = **−0.0340 [−0.1060, +0.0511]** | 40 | 0.433 | 1.47 | 🔴 **v1.6 ckpt is on ONE DISK** (pod2, off-limits) — registry §1.4b flags this explicitly, and `Sayood/flagship-v16-ab-ft` holds **no weights** | **The G-A gates are recorded UNRESOLVED because of this null.** Re-scoring needs the ckpt, and the ckpt is the single most at-risk artifact in the program. **See H3 — this is a stranding item, not just a stats item.** |

**Not reopened, and said plainly:**
- **`2026-07-25-h2-e0-e1` (the L1_gate refutation)** is **NOT** an unpowered null. Its held-out lift is
  `1.1603 [0.9975, 1.3272]` at **n = 2 159 episode clusters** against a pre-registered **1.5×** bar.
  Difference-form Δ = +0.0371 [−0.0006, +0.0759] (prox 0.970) — so it *is* one hair from separating
  from **zero**, but it is **nowhere near the bar**. More power confirms a small real effect that still
  fails the criterion. **The refutation stands.** (Recorded so nobody re-litigates it.)
- **`scaleab_refc-base vs refc-xl`** — prox **0.000–0.050**. The scaling ladder's tie is a **genuine**
  tie, not a power artifact. Do not re-run it.

### H1.1a ⭐ The cleanest demonstration that these nulls are POWER-limited, not effect-limited

Two artifacts measure **essentially the same 0.20 m closed-loop effect**, on the **same 12 held-out
episodes / 265 windows**, with the **same estimator** — and they disagree on the verdict:

| what | Δ ADE@2s | CI95 | `separated` | artifact |
|---|---:|---|---|---|
| **B − A** — re-plan-on-imagination vs single-shot | **−0.2130** | **[−0.3413, −0.0527]** | ✅ **TRUE** — published as *"the proof"* | `…/2026-07-22-imagination-closedloop-proof/closedloop_flagship-30k_imagination-proof.json` |
| **C − A** — WM-MPPI vs single-shot | **−0.2016** | **[−0.3616, +0.0154]** | ❌ **FALSE** — never surfaced | `…/2026-07-23-freefloor-rung3-wm-mpc/wm_mpc_result.json` |

**A 5 % difference in effect size flips the verdict.** The second interval misses zero by **0.0154 m**.
⇒ At this n, *whether a real effect is visible is decided by noise, not by the effect.* This is the same
corpus, the same model, the same statistic — and it is **MEASURED by me from the two raw JSONs**
(class MEASURED · tier **CONFIRMED**, two independent artifacts).

**Consequence for how the program reads its own closed-loop record:** the imagination thesis rests on
the row that happened to separate. The row that did not separate was published as **"TIE — do NOT
promote rung 3"**. Those two sentences are drawn from the same 0.20 m.


### H1.1b ⭐ The cross-arm ranking — the program does not currently know its own order

`…/2026-07-25-jack-blast-radius/jack_recompute.json` re-derived **10 paired comparisons** on the correct
estimator, same 881 windows / 40 episode clusters. **MEASURED, re-read by me from the raw JSON.** Seven
separate cleanly. Three do not — and two of those three are the leaderboard's top rows:

| pair (`a − b`, lower = better for `b`) | Δ | CI95 | verdict at n=40 | **prox** | **prox@600** |
|---|---:|---|---|---:|---:|
| `flagship-v16-ab-ft` − `flagship-30k` | +0.0104 | [−0.0888, +0.1147] | tie | 0.102 | 0.35 → **still a tie** |
| ⭐ **`refc-xl-30k` − `flagship-30k`** | **+0.0443** | **[−0.0544, +0.1465]** | **"they tie"** | **0.441** | **1.50 → would SEPARATE** |
| ⭐ **`refc-base-30k` − `flagship-30k`** | **+0.0457** | **[−0.0555, +0.1506]** | **"they tie"** | **0.443** | **1.51 → would SEPARATE** |
| `refc-xl-30k` − `refc-base-30k` | −0.0013 | [−0.0316, +0.0281] | tie | 0.044 | 0.15 → **genuinely flat** |

**Read the four rows together and they tell a coherent story the program has not stated:**
the two REF-C arms are **indistinguishable from each other** (prox 0.044 — a real tie, and it agrees
with the scaling-ladder verdict), while **both sit ~0.045 m behind flagship v1 with intervals that
would clear zero at n=600.** v1.6 is flat against v1 either way.

⇒ **The plausible n=600 outcome is: `flagship-v1 > {REF-C-XL ≈ REF-C-base}`, separated** — where the
registry currently records *"a difference of split-means; the paired test says not separated — they
tie."* **The program's deployed model may genuinely be its best model, and it does not know it.**

⚠️ **PROVISIONAL, and the caveat is load-bearing:** two arms at prox ≈ 0.44 need the *full* mean ×3.4
shrinkage to clear, so this is the least robust of the H1 projections — it is **exactly the case where a
moving point estimate would kill it**. It is listed here because a rank order between the program's top
three arms is worth an eval run **whatever the answer**, not because the answer is known.


## H1.2 The re-scoring economics — why this is the cheapest win in the program

| | |
|---|---|
| what is needed | re-run `eval_*.py` at `--episodes 600` on checkpoints **we already hold**, on a corpus **we already hold** |
| precedent | `…/2026-07-26-pod2-eval-host/artifacts/RESULT_v1_600ep.json` — **already done once, for v1** |
| GPU cost | **0 training hours.** Eval only. |
| arms with ≥2 durable copies (safe to re-score any time) | `flagship-30k` (v1) · `refc-xl-30k` · `refc-base-30k` · `refa-dinov2` — all on **HF gated + eval pod + source pod** |
| arms on ONE disk (⚠️ re-score is also a **rescue**) | 🔴 **`flagship-v16-ab-ft`** (pod2 only, HF repo empty) · `flagship-v4.1-10k` (registry: *"single pod disk — HF-back it"*) · `flagship-v3enc-10k` (registry: *"DO NOT RECYCLE tanitad-pod"*) · `dynenc-branchB` (pod3 only, HF push **blocked** by the safety classifier) |
| **local free win, needs no pod at all** | **27 `windows_*.pt` per-window dumps sit in `taniteval/results/`.** They permit **re-pairing any two arms at n=40 on the correct estimator, on this laptop, in minutes.** That is how `v16_vs_v1_paired_bootstrap.json` and `jack_recompute.json` were produced. It cannot give n=600 — but it can re-adjudicate every `_jack` number for free. |

## H1.3 ⚠️ The inversion nobody has stated — half these nulls are DANGEROUS, not lost value

The brief frames H1 as *"real effects we discarded"*. That is right for **treatment** comparisons. It is
**exactly backwards** for the program's **firewall / negative-control / leakage / guardrail** checks,
where a null is the *desired* verdict:

- `4brain-s3` / `s3-decision-grade`: `paired_leak_B2_minus_B1` (**a leak probe**) prox 0.835 → **2.84**
- `4brain-gates` S1: `firewall.NOGOAL.blind_vs_majority_paired`, `firewall.E/H.blind_vs_majority_paired`
  — the blind-baseline firewall that decides whether a task is admissible at all, at **n = 6–20 clusters**
- `lead-state-gate`: `paired_mae_A_minus_B_shuf` — the **shuffle control**, prox 0.903 → 3.07

**For these, "not separated at n=40" is not a refuted leak — it is a leak we could not see.** The S3
firewall ADMITTED its tasks on nulls that project to ~2.8× separation. If those leaks are real, every
downstream S3 number is contaminated. **This should be re-adjudicated BEFORE the treatment effects**,
because it can only ever remove results, and removing a false result is worth more than adding a true one.

**Concrete: 5 firewall/control nulls with prox > 0.8 at n ≤ 126.** They are listed in
`harvest_index.json` under `h1.firewall_inversion`.

## H1.4 Nulls that are not results at all — the `_jack` residue

**21 `separated: false` nodes were produced by the deprecated `overlapping_holdout_se` / `_jack`
estimator.** Per `CLAUDE.md` and the 07-25 retraction, this biases **both** the interval (1.107–3.100×)
and the **point estimate** (−6.67 %…+11.69 %, up to ×−4.15 **including sign flips**). They are all in
`hierarchy.py` output:

| file | node | effect | half-width | note |
|---|---|---|---:|---:|---|
| `…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-30k.json` | `seam_intent_to_operative.delta_cos_real_vs_none` | −0.2384 | 0.0034 | prox **70.1** — a "not separated" verdict on an effect **70× its own half-width**. This is not a null, it is a **broken emitter**. |
| `…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-v4.2b-dryrun.json` | same | −0.2199 | 0.0047 | prox 46.8 |
| `…/2026-07-23-v4-gate-emitters/artifacts/hierarchy_flagship-30k_v1.json` | same | −0.2464 | 0.0071 | prox 34.7 |
| (+18 more, all `hierarchy.py`) | | | | |

⚠️ These three are the most suspicious nodes in the entire sweep and I flag them as **an instrument
defect, not a finding**: `_jack`'s `separated` was **one-sided** (documented in the 07-25 retraction), so
a large *negative* effect renders `separated: false` **by construction**. The `intent→operative` seam is
recorded as **harmful** in the registry — consistent with a large negative delta — but the JSON says
"not separated", which is a false statement about a 70-half-width effect. **Anyone reading these files
directly (rather than the report) gets the wrong answer.** Fix: re-emit through
`paired_episode_cluster_bootstrap` and emit `separated_positive` alongside `separated`.
*(`hierarchy.py` was migrated per the 07-25 retraction — but **these committed artifacts were not
re-emitted**, so the stale files are still the primary source anyone would grep. See H3.)*

## H1.5 Ranked re-adjudication programme (this is the actionable output)

| pri | batch | arms | n | cost | why first |
|---|---|---|---|---|---|
| **1** | **Firewall/control re-adjudication** (H1.3) — S3 leak probes, S1 blind-baseline firewall, lead-gate shuffle | v1, S3 arms | 600 (or larger clip draw) | eval-only | Can only **remove** false results. Contamination is upstream of everything. |
| **2** | **Re-emit the 21 `_jack` hierarchy nodes** | v1, v4.2b | 40 | **local, CPU, minutes** — the `windows_*.pt` are here | Costs nothing. Removes three files that state the opposite of the truth. |
| **3** | **Closed-loop 12-ep suites → 40 then 600** (rows 1, 6, 7, 8, 9, 13) | v1, REF-C base/XL | 12→600 | eval-only, ckpts have 3 copies | **The densest cluster of near-separations in the program (466 nulls at n≈12).** The closed-loop direction has been declared closed/bound **five times** and reopened by cheap follow-ups **five times** (RETRACTION_LOG meta-pattern). This is the sixth cheap follow-up and it is a power check. |
| **4** | **v1 vs REF-C-XL at n=600** (row 19) | both, 3 copies each | 600 | eval-only | Would give the program a **rank order between its two best arms**, which it currently does not have. |
| **5** | **Rescue-and-re-score v1.6** (row 20) | 🔴 one disk | 600 | needs pod2 free | Rescue is the point; the number is a bonus. |

## H1.6 Prose-only nulls — closing method limit #1

The JSON sweep cannot see a verdict that exists only as a sentence. I grepped all 222 `.md` files for
null language (87 files matched) and read the `*/Research/` layer, which is where verdicts live without
a JSON sibling. What the JSON sweep **missed**, and it is a short list:

| null, as written | where | n | prox (from the quoted interval) | why it matters |
|---|---|---:|---:|---|
| *"[−1.4273, +0.4431] — **not separated from zero**. v3enc has **no systematic longitudinal bias**"* | `Architecture & Inference/Research/2026-07-21-flagship-v3enc-postmortem.md:306` | n/a in text | ≈ **0.53** | This is a **post-mortem finding on an arm the gate sent to RESTART**, and it is one of the exculpatory readings. Its checkpoint carries a registry warning: *"DO NOT RECYCLE `tanitad-pod`… the only 10 k state that will ever exist."* |
| *"3-class accuracy 0.7816 vs 0.7586 (McNemar p = 0.625, **not separated**)"* | `Data Engineering/Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md:227` | — | — | ⚠️ **NOT a bootstrap null — a McNemar test.** The ×3.4 shrinkage argument does **not** transfer to it; it needs more *labelled items*, not more episodes. Recorded so nobody applies the H1 projection to it. |
| *"FDE@2s **not separated**; self-referential"* | `Architecture & Inference/Research/2026-07-23-planner-is-the-bottleneck.md:98` and `…/2026-07-22-closed-loop-robustness-and-imagination.md:99` | 12 | 0.862 | This is the **prose carrier of H1.1 row 8** — the same FDE leg, propagated into two synthesis documents as a caveat on the imagination thesis. **A single re-score at higher n would remove a caveat from two synthesis docs at once.** |
| *"DAgger-FT **ties open-loop** (CI includes 0) ⇒ on-policy state coverage is **not** the bottleneck"* | `Architecture & Inference/Research/2026-07-23-planner-is-the-bottleneck.md:245` | 12 | see H1.1 row 13 | The prose carrier of row 13, and note the inference it licenses: a **capability-level conclusion** ("on-policy coverage is not the bottleneck") drawn from an n=12 tie. |
| REF-C **medium-vs-XL** scaling: three nulls quoted in the text with intervals | `Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md:19-21` | 40 | **0.044 / 0.050 / 0.000** | ✅ **Genuine ties, not power artifacts** — the JSON sweep agrees. Recorded to close the question: **do not re-run the scaling ladder.** |

**Conclusion on limit #1:** the prose layer adds **no new near-separations** — every prose null I found
either (a) is the text form of a JSON node already ranked above, (b) is a genuinely flat tie, or (c) is a
**different test** (McNemar) to which the H1 projection does not apply. The JSON sweep is therefore
**not materially incomplete** for ranking purposes. ⚠️ This is a **spot check of the Research layer, not
an exhaustive read of 222 files** — I am claiming "no new near-separations found where I looked", not
"none exist".

---

# H5 — results that are levers nobody connected

*Findings produced by one stream that answer a **different** stream's open question. Ranked by
(value of the question answered) ÷ (cost of connecting it). This is the highest-value output after H1.*

**The open questions these are matched against** (from `BOOST_PROGRAM.md` §3.3, §4, §5):
`Q-BarB` **`wm_canary_ade_2s` must fall 2.07× — NO LEVER IDENTIFIED, currently UNOWNED** ·
`Q-BarA` selector must recover ≥70.8 % of 0.6058 m waste · `Q-CL` no admissible closed-loop horizon ·
`Q-TOPO` the strategic-brain topology must come from AlpaSim or an external corpus (PhysicalAI has no
map — settled at five probes) · `Q-HP4` compositional generalisation to unseen junction topologies.

| # | the finding (its stream) | the question it answers (a **different** stream) | why the connection was missed | cost to connect | class · tier |
|---|---|---|---|---|---|
| **1** | ⭐ **Argoverse 2 is credential-free AND its lane graph is byte-verified.** `successors` + `predecessors` + `left/right_neighbor_id` + `is_intersection` on **7,692 lane segments across 85 maps, 100 % field presence**; anonymous `s3://argoverse` returns HTTP 200 unsigned; the publisher's own guide says no AWS account is needed. *(`…/2026-07-26-credential-free-lanegraph/LANEGRAPH_ALTERNATIVES.md` §1, §4; evidence JSONs in `evidence/`)* | **`Q-TOPO`.** The program's standing position — reinforced by five probes and a PI-approved-then-retracted ZOD recommendation — is that a lane graph must come from AlpaSim or an external corpus, and that the external candidates were all gated or empty. **AV2 is neither.** It is also the exact property ZOD was promoted for and does not have. | The AV2 work landed **the same day** as the ZOD retraction and the Overture pivot. Three lane-graph threads ran in parallel; the winner is in a *third* directory from both. | ⚠️ **MEASURED BY ME, two probes:** `stack/tanitad/data/argoverse2.py` **and** `stack/tests/test_argoverse2.py` exist at HEAD (`git ls-files`), the licence is in `SOURCE_REGISTRY` — but **`stack/scripts/ingest_argoverse2.py` is ABSENT** (`test -f`) and **nothing in production imports the adapter** (`grep`: only `lake/schema.py` + tests). ⇒ **one ~40-line driver mirroring `ingest_nuscenes.py`, plus the ~147 MiB pull already on the PI's decision list (BOOST §5.5).** | MEASURED (mine) · **CONFIRMED** (adapter presence and driver absence each verified by two independent tools) |
| **2** | ⭐ **We already OWN a map and reactive agents, and have switched on neither — for weeks.** AlpaSim scenes embed `trajdata.VectorMap` (**130–472 lane polygons + 130–393 road edges + wait-lines per scene**, loadable at inference), and `trafficsim` (SMART/CAT-K, **Apache-2.0, in-tree, already on the pod**) has **never once been enabled**. *(`…/2026-07-26-alpasim-consolidation/ALPASIM_STATE.md`; `…/2026-07-26-4brain-dominance-program/4BRAIN_DOMINANCE_PROGRAM.md` §PC4 + §4.3)* | **`Q-TOPO` and `Q-HP4` simultaneously**, plus the T1–T4 tactical gates. A strategic problem needs a topology to choose over; a tactical problem needs an agent to choose against. **Both assets are owned.** | The program's own words: *"We have owned both for weeks and switched on neither."* The blocking belief was *"we have no map"* — true of **PhysicalAI**, false of **AlpaSim**, and the two were conflated. This is the **C2 class (absence from a single probe) applied to a capability rather than a file.** | **Zero GPU.** The 4-brain program already scoped it: a **~1 h read-only VectorMap connectivity probe on the eval pod** that *"gates S1, S2, S4 and HP-4 — four of the nine problems"*, and a 1–3 d `trafficsim` one-scene rollout. **The eval pod is listed as free.** | **MEASURED (mine) · CONFIRMED** for the *"present but off"* half — second probe at HEAD: `…/2026-07-22-alpasim-closedloop-evalpod/RUN_RECIPE.md:26` reads **"trafficsim (disabled by default) — non-ego actors"**, and `pyproject_pared.toml` shows `alpasim-trafficsim` is a **workspace package installed by `uv sync --extra core`**. So it is built, installed, and off by a default flag. ⚠️ The *map* half stays INHERITED · PROVISIONAL: the prior probe (`gate0_prereq_probe.json`) measured **counts only and its trajectory read ERRORED**, so *"the graph has connectivity"* is **NOT established** — that is exactly what the 1 h probe settles. |
| **3** | ⭐ **The deployed tick is 74.24 GFLOP, not 401.9 — and its DRAM bytes are 95.5 % rollout, so every FLOP-trading lever is worth ≈ 0 %.** On Orin the rollout sits **28×** below machine balance, on Thor **129×**. *(`…/2026-07-26-orin-thor-optimization/ORIN_THOR_STATE_AND_PLAN.md` headline)* | **The H2 sensor-attention stream's entire cost model.** H2's surviving quotable result is an **encoder-compute** saving (*"0.67 % residual need-rate ⇒ 1.007 cameras/frame ⇒ 84.8–85.6 % saved vs always-on-7"*), and its central argument was **selective activation vs widening the crop, priced in pixels and tokens**. On the actual deployment target **that currency is ≈ free**. | The two streams never met. H2 has **already been retracted once for a costing error** in this exact shape (C3, 07-26: *coverage silently treated as cost*). This is the **same error one level up** — and the right currency was measured by another agent the next day. | **Zero compute.** Re-derive H2's cost table in **DRAM bytes** using the Orin per-component budget. Both artifacts are on disk. ⚠️ The sign is **not** obviously "H2 is dead": cheap FLOPs may make an occasional second encode nearly free, which *strengthens* selective activation. **The point is that the quoted number is in the wrong unit, not that it is wrong.** | MEASURED (each stream, ours) · **PROVISIONAL** — the *cross-application* is my inference and has not been computed. |
| **4** | ⭐⭐ **TWO INDEPENDENT STREAMS now support the same Bar-B lever: viewpoint / off-path augmentation.** (a) P1/C14: the yaw warp is **geometrically EXACT** (`max abs(ΔH) = 0.000e+00` over 30 conditions) ⇒ *"roughly half of what the envelope measures is OUR ARM'S OOD SENSITIVITY… the lever is TRAINING-TIME OFF-PATH AUGMENTATION, not rendering"*. (b) The own-dynamics-encoder line, **independently and on a different axis**: the same encoder family reads **in-domain rig-A speed R² ≈ 0.80–0.85** and collapses to **strongly negative cross-rig**, and the multi-rig arm **REFUTED data-diversity** as the cause — *"the collapse is REPRESENTATIONAL"*. *(`RETRACTION_LOG.md` 07-26 C14; `…/2026-07-22-own-dynamics-encoder/RESULTS_camcond.md`; `…/2026-07-24-branchb-transfer-eval/`)* | **`Q-BarB` — the bar with NO identified lever, explicitly UNOWNED in `BOOST_PROGRAM.md` §3.3.** `wm_canary_ade_2s` is world-model fidelity under self-rollout, i.e. exactly the regime where the state drifts off the training manifold. | The dynenc stream is filed under **"REFUTED, both branches spent"** and was closed as a *failure*. Its **negative** result (conditioning does not fix it; **data diversity does not fix it either**) is a **positive constraint on the fix**: the remaining lever is neither architecture-conditioning nor more corpus — which is what leaves augmentation standing. **A refuted stream's constraint is not the same as a dead stream.** | Design-only to state; the experiment is a training-arm decision. **But it is the first named Bar-B candidate, and it arrives with two independent supports rather than one** — which is the M1 CONFIRMED bar. | INHERITED (both measurements) · the **coincidence** is my inference · **PROVISIONAL** — a *hypothesis for a lever*, not a measured lever. It must not enter a kill conjunction. |
| **5** | ⚠️ **The `wm_canary` descent rate that `BOOST_PROGRAM.md` §3.3 uses to argue Bar B is hard is a TRAINER-LOG number.** BOOST quotes *"−21.6 % per 20k steps"*, sourced from `…/2026-07-26-v4-restart-lever/raw/lambda_verdict.json` (`canary_8_10k` **1.4900** → `canary_26_30k` **1.1689**). The **eval-grade** pair on the pinned 881 windows is **2.0739 @15k → 1.1409 @30k = −45.0 % over 15,000 steps** — and the trainer series reads **1.4900 at step 8–10k while eval reads 2.0739 at 15k**, i.e. **lower at an earlier step: they are not the same statistic.** *(MEASURED by me from the three raw JSONs.)* | **`Q-BarB`'s difficulty estimate**, which is doing real work in the recovery plan's recommendation *not* to restart v4. | `RETRACTION_LOG` **C1** says only `eval_*.py` output is quotable; the 07-25 entry adds *"a metric NAME is not a metric DEFINITION"*. Both apply here and neither was applied. The restart-lever report itself **correctly refused to extrapolate** — the rate was picked up downstream and used as if it were the eval rate. | **Zero.** Two eval points already exist. ⛔ **I do NOT claim Bar B is reachable** — two points fit no rate, and the program forbids extrapolation without window, R² and n. The claim is narrower and MEASURED: **the number in the recovery plan is from the wrong instrument, and a third eval point would settle it.** Re-scoring a preserved intermediate v4 ckpt is **eval-only**. | MEASURED (mine, three raw JSONs) · **CONFIRMED** (level *and* rate disagree — two independent checks) |
| **6** | ⭐ **HP-4 is ~17 scenes away, not a corpus rebuild away** — 0 of 23 topology classes reach the ≥40-cluster bar today; best is `S|S` at 38 scenes. *(`…/2026-07-26-vectormap-corridor/VECTORMAP_CORRIDOR.md` §5)* | **`Q-HP4`**, and it converts an "impossible" into a **procurement item**. It also converts BOOST §5.6's *"+17 scenes for HP-4"* from a bare ask into a justified one. | Filed under the headline **⛔ "HP-4 is NOT runnable today"**, which reads like a closure; the actionable *"~17 scenes"* sits 30 lines below it. **A blocker with a price is not a blocker.** | Scene selection, which `ALPASIM_STATE.md` records as **already solved** (the balanced-suite builder *"turned 0 roundabout scenes into 8"*). Combine with row 1 (AV2's 85 maps) for topology classes AlpaSim cannot supply. | INHERITED · PROVISIONAL (the "17" is `ESTIMATED` in its own source — I did not re-derive it) |
| **7** | **`obstacle.offline` — 3D agent tracks on ~97 % of the corpus, 87,481 cuboids, 10 dynamic classes — was refused on a gate that tested something else.** The `lead-state-gate` measured *"does lead state improve prediction of the **EGO's** 2 s along-track displacement"* → **+1.16 % [−0.92, +3.19]**, and correctly refused a 12.4 GB ingest. | **`Q-BarB`.** `wm_canary_ade_2s` is **world-model** fidelity — how well the model predicts the **scene's** future latent, not the ego's waypoint. **The gate's null does not speak to it.** A world model asked to imagine a dynamic scene without ever seeing agent state is being asked to predict what it cannot represent. | The gate's refusal was **right and is not reopened here** — but it has been carried forward as *"agent state is refuted"* in general, when its pre-registration scoped it to *ego 2 s longitudinal prediction*. This is the **C12 class** (a composite null blamed on the wrong half) at the level of a whole capability. | ⚠️ **Do NOT re-ingest on this reasoning alone** — that is exactly the mechanism-without-measurement the original gate refused (logged C3, 07-21). The cheap step is a **second gate with a WM-fidelity target**, on the **1.1 GB slice already pulled** for the first gate. **~0 GPU**, and the precedent exists. | INHERITED (the gate) · **the re-scoping is my inference · PROVISIONAL · HYPOTHESIS for the lever** |
| **8** | **The n=600 re-scoring path is already built and proven, not a new capability.** `…/2026-07-26-pod2-eval-host/artifacts/RESULT_v1_600ep.json` exists; the 600-build is a **MEASURED order-preserving superset** of the canonical 40 (`prefix_disjointness_result.json`), so parity is not violated. | **All of H1.** The 789-null backlog is blocked on *nothing* except eval-pod time. | The pod2 stand-up report's headline is about the eval **host** (staleness, missing `corridor.py`, the PEP-701 SyntaxError); the **reusable 600-ep protocol** is a by-product, recorded in registry §1.2a. | **Zero engineering.** Queue arms behind the Bar-A experiment. | MEASURED · CONFIRMED (registry §1.2a publishes it with both artifacts) |
| **9** | **H18 grounding dominance is the hierarchy's strongest positive AND it is immune to the estimator problem that killed the others.** Paired Δ **+2.9568 m** (corrected **UP** from +2.6979 under the fix) — un-separating it would need an **8.65× interval widening against a worst-ever-measured 2.06×**. | **The hierarchy question**, which currently reads *"0 of 3 seams load-bearing"* after the `ctx→tactical` retraction. **0/3 plus one 8.65×-robust dominance result is a different picture from 0/3.** | The `_jack` retraction (07-25) is dramatic and its headline is the **0/3**. The same entry's closing lines record that H18 **strengthened** — it moved up, not down. **A retraction's good news travels worse than its bad news.** | Zero — it is already measured. It needs to be **stated together with the 0/3**, in the registry and in BOOST, or the program keeps reasoning from half its evidence. | INHERITED (RETRACTION_LOG 07-25 + 4BRAIN §E1/H18) · PROVISIONAL |
| **10** | **PC3 is unblocked in code and unmeasured on any real arm** — `corridor.from_windows` runs on any archived arm, but **no archived arm has `pred_dense`**, so it has never been evaluated. *(`4BRAIN_DOMINANCE_PROGRAM.md` §§4.3, 5)* | **`Q-CL`** — the closed-loop measurability stream (S-1), the program's #1 blocker. A per-window dense-prediction dump is exactly what horizon-capable corridor scoring needs. | It is a **one-flag emitter change** at eval time, filed inside a 350-line program plan under a small marker. | One flag on the next eval run. **Cheapest row here.** | INHERITED · PROVISIONAL |

## H5.1 The pattern across these ten

Eight of the ten share one failure shape: *a stream produced a fact, filed it under its own question,
and the fact answers somebody else's.* Three sub-shapes recur and are preventable:

1. **A refuted stream still produces positive constraints** (rows 4, 7). "Branch A and B refuted" was
   filed as a dead end; what it *measured* is that neither conditioning nor data diversity is the fix —
   a **narrowing** of the Bar-B search space, and the most useful thing anyone has said about a bar
   that has no lever.
2. **A blocker with a price is not a blocker** (rows 6, 1). "HP-4 is NOT runnable" and "no lane graph
   exists" each had a number attached ~30 lines further down.
3. **A retraction's good news does not travel** (rows 9, 5). The 07-25 `_jack` entry corrected H18
   **upward**, and the 07-26 restart-lever report **refused to extrapolate** the canary — both careful,
   both correct, and both lost the moment the headline moved on.

**⚠️ The honest caveat on this whole section.** Every row is a *connection I am proposing*, not a
measurement I made. Rows 1, 5 and 8 carry MEASURED facts I verified myself with two probes each. Rows
2, 3, 4, 6, 7, 9 and 10 are **INHERITED · PROVISIONAL** — hypotheses about where value is. Under
`BOOST_PROGRAM` M1 **none of them may decide a GPU-day until an independent path reproduces them.**
The cheapest ones (rows 2, 8, 10 — all ≈ zero cost) should be run first precisely because running them
is cheaper than arguing about them.

---

# H3 — stranded integrations → **full inventory in `H3_STRANDED_INTEGRATIONS.md` / `h3_stranded.json`**

**23 distinct claims chased down and re-verified against HEAD** (not taken from the reports that raised
them — the brief's five known examples were a starting point, not the answer):

| | count | note |
|---|---:|---|
| **STILL OPEN** | **12** (`S-01`…`S-12`) | 11 MEASURED·CONFIRMED against HEAD; `S-12` (v1.6 on one disk) unverifiable without pod/HF access |
| **ALREADY DONE — the escalation is stale** | **11** (`D-01`…`D-11`) | mostly fixed **the same day**, by a sibling agent or the next commit |
| resolved-by-decision / moot | 2 (`D-12`, `D-13`) | `D-12` was never a bug; `D-13`'s premise was retracted |

⭐ **The finding that matters most, and I verified it independently of the H3 agent:**

> **`taniteval/taniteval/clhorizon.py:509` calls `_data.load_frames(...)`, which returns `RawEp` —
> whose constructor sets `self.feats = ep.frames` and defines **no `.frames` attribute** (`data.py:216-222`) —
> and hands it to `corridor_rollout`, whose default `frames_of` reads `ep.frames`. `_data.load_raw`
> exists at `data.py:131` and is what the committed gate driver used.**
>
> ⇒ **The module written specifically so *"the co-primary is not stranded behind a driver in
> `incoming/`"* raises on its first rollout step and has evidently never been executed end-to-end.**
> One line. **MEASURED · CONFIRMED — two agents, two independent code reads.**

**Three more STILL-OPEN rows deserve to be read as program-level, not clerical:**

1. **`S-03` `planner_p2.py` is the ONE sibling that missed the `_jack` migration** — `hierarchy.py`,
   `closedloop.py` and `bench.py` all now name `paired_episode_cluster_bootstrap` as `primary_estimator`;
   `planner_p2.py` has **zero** `episode_cluster_bootstrap` call sites, and `G1_pass`/`G4_pass` — the
   **CEM-planner-beats-supervised-head** verdicts — are computed directly off `_jack`. Given the measured
   ×−4.15 **sign-flipping** bias, **those verdicts may be reading the wrong sign.** *(This is H1.4's
   problem in a second location, and it is why action 5 ranks where it does.)*
   ⭐ **Sharpened by my own read, and it makes the row worse: the file DOCUMENTS ITS OWN DEFECT and
   nobody acted.** `planner_p2.py:382-385` — the docstring of `_jack_paired` itself reads
   *"**DEPRECATED estimator (not a jackknife). Prefer `ci.paired_episode_cluster_bootstrap`** — these
   arms share windows."* That is the file's **only** mention of the correct estimator; the function is
   still what computes `G1_pass`/`G4_pass` (lines 399, 442, 570). ⚠️ **This is precisely the
   10-day-README failure the Agent Operating Standard rule 3 exists to prevent** — a correct
   instruction written into the artifact it applies to, where nothing has to read it.
   ⚠️ **Second-order, and it is the point-estimate half of the problem:** `_jack_scalar` (lines 373-378)
   returns `mean = np.mean(split_means)` — the **`heldout` mean-of-split-means**, which `CLAUDE.md`
   flags as biased **−6.67 % to +11.69 %** against the `full_set` mean. So P2's *central values*, not
   only its intervals, are the deprecated statistic. **MEASURED (mine, code read at HEAD) · CONFIRMED.**
2. **`S-10` a committed artifact still publishes FALSE verdict strings — and I re-verified it myself,
   finding it is BROADER than reported.** Walking every `EXTRAPOLATION_VERDICT` node in
   `…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json` (MEASURED, mine):

   | node | verdict string | actual frac. of windows OUT of envelope |
   |---|---|---:|
   | `all_windows.185.{overall,junction,longitudinal,other}` | ✅ *"EXTRAPOLATION — NOT a measurement at this horizon"* | 0.9024 / **1.0000** / 0.9444 / 0.8235 |
   | `paired_common_start.185.{overall,junction,longitudinal,other}` | ❌ *"within the measured envelope on average"* | 0.9024 / **1.0000** / 0.9444 / 0.8235 |
   | `paired_common_start.20.{overall,junction,longitudinal,other}` | ❌ *"within the measured envelope on average"* | 0.0488 / **0.3333** / 0.0 / 0.0 |

   ⇒ **8 false verdict strings, not 4** — the K=20 block is wrong too (junction: a third of windows
   outside, string says "within"). The `all_windows` half of the **same file** is correct, so a reader
   who lands in the wrong half gets **the exact opposite of the truth on the identical windows**.
   MEASURED · **CONFIRMED** (two agents, two independent walks). Original description below: `fix_ood_verdict.py` only
   walks `d["all_windows"]`, never `d["paired_common_start"]` — so in `corridor_v4_30k_K185.json` all
   four strata still read *`"EXTRAPOLATION_VERDICT": "within the measured envelope on average"`* beside
   out-of-envelope fractions of **0.90 / 1.00 / 0.94 / 0.82**. The sibling node in the *same file* is
   correctly stamped *"EXTRAPOLATION — NOT a measurement."* **Anyone reading the uncorrected half gets
   the opposite of the truth.**
3. **`S-04` the three CUDA-graph-capture prerequisites are the only genuinely multi-day item — 6 days,
   "needs an owner", exact lines still unchanged** (`metric_dynamics.py:241-242`, `predictor.py:112`,
   `predictor.py:118-121`). They block every truthful CUDA-graph latency number on the Orin/Thor axis.

⭐ **The meta-finding, and it is good news that also indicts our publishing speed:** *"unlike the
10-/12-day examples, most of today's stranding is **same-day** — a report escalates at hour N and a
sibling fixes it by hour N+4."* The harvest loop is working. **But it also means several reports written
earlier TODAY are already out of date by the time anyone reads them carefully** — which is
`BOOST_PROGRAM` **C-I** ("we publish at agent-completion speed, not verification speed") appearing in a
new place: not in what we publish, but in what we *re-read*.
⚠️ **Operational consequence:** 11 of 23 escalations were already fixed. **Before acting on ANY
escalation in this repo, re-verify it against HEAD — the base rate of stale escalations is ~48 %.**

---

# H2 — capabilities built but never used → **full inventory in `H2_UNUSED_CAPABILITIES.md` / `h2_unused_capabilities.json`**

**38 capabilities** catalogued: what it does · who imports it · gate/report wiring · does it have a test ·
**does it actually run** (executed where possible, not assumed) · the cheapest step to put it to work.

## The four that are not "unused capability" but **live defect**

| | finding | why it is worse than idle code |
|---|---|---|
| **`H15Meter`** *(`…/2026-07-15-h15-logging-fidelity/`)* | `stack/scripts/train_flagship4b.py:522` still logs `log["h15"] = float(loss_h15.item())` — **the single LAST micro-batch**, where the module's own docstring measures that this misreports the accumulated-batch value **~46 % of the time**. | **This is wrong in every training log the program is producing right now**, including the running v2-corpus arm. Fix is ~3 lines. |
| **`blind_conditioning_baseline`** *(`taniteval/blind_baseline.py`)* | Exists, 14 tests, **already produced** `blind_firewall_route_target.json` in which **all three route-target labelers score `CIRCULAR`** — and **nothing calls `assert_registered` from `run_gate.py`**. `PRECONDITIONS_IMPLEMENTED.md` §7 item 6 says so in as many words. | ⭐⭐ **This is H1.3's inversion with a name.** The firewall that would catch circular targets is **built and not wired**, and it is so undiscoverable that **two separate `incoming/` streams re-implemented inferior copies of it in the same week** (`2026-07-26-4brain-gates/blind_conditioning_baseline.py`, `2026-07-26-4brain-s3/s3_blind_baseline.py`) rather than finding it. **The under-powered firewall nulls in H1.3 came out of those re-implementations.** |
| **`validate_operative_inputs`** *(`…/2026-07-09-models-predictor-failfast/`)* | `stack/tanitad/models/predictor.py:101` still uses a bare `assert w == self.cfg.window` — **stripped entirely under `python -O`**. The `ValueError` version exists and is tested. | A silent shape-corruption path in core forward-pass code, directly against `CLAUDE.md` rule 5 (*fail loud*). |
| **`probe_r2_ci.py`** *(`…/2026-07-21-gate-secondary-probe-audit/`)* | `encoder_speed_probe_r2` — **which was used to justify RESTARTING flagship-v3enc, a GPU-week decision** — is still computed by the unaudited original: **a single deterministic split with in-sample lambda selection**, never CI-audited. The split-robust re-estimator exists and is tested. | A GPU-week was spent on a number whose interval has never been computed. |

## The two that make our own documents wrong

1. ⭐ **`tools/safe_commit.py` is the sanctioned commit path and `CLAUDE.md` does not mention it.**
   Built **the same day** as CLAUDE.md's Git-hygiene section (rule written 00:41, tool created 16:07 by a
   later commit **specifically to mechanize it**), 127/127 tests green, `--print-index` verified against
   this live repo — and **nothing calls it**. CLAUDE.md still instructs every agent to run raw
   `git commit -F`, i.e. the procedure the tool exists to make safe. *(H2 rates this its one
   **DECISION-GRADE** row; the fix is a **doc edit**, <1 h.)*
2. ⚠️ **The orthogonality instrument is at 16 days, not 10.** `CLAUDE.md:57-58` and the Agent Operating
   Standard both cite it as *"sat unmerged for **10 days**"*; H2 re-verified the 8/8 test suite and traced
   **two** separate escalations by name (2026-07-10 original, 2026-07-17 independent re-verification
   reproducing the number exactly, 2026-07-18 run on `flagship-speed@19k` → `NOT-YET-ADMISSIBLE`). The
   second escalating agent withdrew their own duplicate draft and wrote it *"should not stay stranded a
   3rd week."* ⇒ **a second stale number inside `CLAUDE.md`, alongside O-5's.** *(It is stale in the
   harmless direction — the real figure is worse — but it is the same defect.)*

## `tools/registry_lint.py` — the tool for the failure `CLAUDE.md` opens with, not in the gate

H2 **ran it today**: `--self-test` 5/5, and a live sweep of `MODEL_REGISTRY.md` → `PASS (0 errors,
2 warnings)`. It checks pointer drift against raw eval JSON **and** does a **multiline** retracted-claim
sweep over section headers — i.e. exactly the *"best ADE in the program"* stale-headline-for-4-days
incident, including the multiline-evasion sub-lesson the 07-25 retraction recorded. It is **not called by
`tools/ci_gate.py` / `ci.ps1`**. Fix: **3–5 lines**.

## Two instruments whose absence is shaping the hierarchy debate

- **`taniteval/planning.py` has never been executed, once.** `report.py`'s *"02c · Planning"* panel is
  **already built** and reads a `plan_<arm>.json` — and **zero `plan_*.json` exist anywhere in the repo**
  (`git ls-files` and `find` both empty). What it would score is `route_acc_follow`, **`route_skill`**,
  `maneuver_bal_acc`, `tactical_wp_ade`, `goal_latent_cos` — **the exact strategic/tactical decodability
  numbers the hierarchy argument turns on**, with the UI waiting for them.
- ⭐ **`taniteval/strategic_probes.py` (HP-3) has been run twice, by hand, and it FALSIFIED its own
  pre-registered expectation.** The prediction was *"~0 by construction"*; the two manual runs give
  **cross-track 0.56 m and direction score 0.62, CI-separated from chance** on the left branch
  (`hp3_prefix_flagship-30k.json`, `hp3_prefix_refc-base-30k.json`). **Nothing in the automatic per-arm
  battery will ever reproduce this for a new checkpoint** — it is not in `runner.py`.
  ⇒ **This belongs in H5 as much as H2:** it bears directly on the confound `CLAUDE.md` flags as having
  *"nearly designed the hierarchy away"* (`route_skill = 0.0`, `nav_cmd=None`), and it points the *other*
  way. **Class INHERITED · PROVISIONAL** (H2 verified the wiring gap and ran the tests itself — 31/31 —
  but did **not** re-run the HP-3 measurement; treat the 0.56/0.62 as unconfirmed).

## Two complete safety scenarios sitting outside the registry

`stop_arm_gate.py` (school-bus stop-arm violation) and `stationary_lead.py` are **complete, tested, and
absent from `SCENARIO_REGISTRY`**, which H2 verified has **exactly three keys** (`work_zone_phantom`,
`traffic_light_red`, `traffic_light_green`). Their built sibling `work_zone_phantom` **is** registered and
wired into a live CARLA runner. Fix: **an 8-line `ScenarioEntry` each**, mirroring the sibling exactly.

⚠️ **Tier discipline on this section:** H2 marks per row which facts it executed itself (`MEASURED ·
CONFIRMED` — e.g. `registry_lint --self-test`, the `SCENARIO_REGISTRY` key count, `predictor.py:101`,
`planning.py`'s absent outputs) versus which it inherited from its own sub-probes (`PROVISIONAL`).
**Read the source table's evidence column before acting; I have not re-verified H2's rows myself**, with
the exception of the `blind_baseline` ↔ H1.3 cross-link, which is corroborated by my own firewall sweep.

---

# ⚠️ FINDINGS THAT ARE LIVE RIGHT NOW — found by the orchestrating agent, outside H1–H5

*These surfaced while cross-reading `LOOP_STATE.md` against `CLAUDE.md`, `RETRACTION_LOG.md` and HEAD.
They are separated out because two of them are **about to cost something**, today, unattended.*

## O-1 🔴 A live standing directive commissions a probe that is already complete at FIVE locations

**The drumbeat acts on `LOOP_STATE.md`'s standing directives WITHOUT re-asking** — that is stated in the
file's own header. Directive **D-A** currently reads:

> *"…then intervention #3 (drivable-corridor channel, gated on a **~1-day probe of all 36 PhysicalAI
> features** for map/lane data — **the 'no HD map' claim has never been second-probed** and its sibling
> 'no agent boxes' was already retracted as C2)."*

**That probe has run, it is complete, and it is committed.** `…/2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md`
establishes at **five independent locations** — the dataset's own `features.csv` (exactly 36 rows, no
map/lane/light/junction row) · the HF tree API L1→L3 · the dataset card verbatim (*"we do not include
open maps data"*) · the `obstacle.offline` class enum over 87,481 rows (10 classes, **all dynamic
agents**) · both tagged revisions — that **there is no map, lane geometry, lane graph, routable topology,
junction annotation, roundabout label, traffic-light feature or route/goal signal in PhysicalAI-AV.**
`CLAUDE.md` already carries the consequence in binding form: ***"Stop re-asking."***

| | |
|---|---|
| what it would cost | **~1 pod-day**, unattended, on a fleet where all three pods are busy |
| verified by me | `git ls-files …/2026-07-26-physicalai-feature-probe/` → **19 files tracked** (so the report's own *"NOT staged (per brief)"* line is **stale** — it was staged later; I checked before asserting, because the mirror-image error is a logged retraction) |
| the fix | **one edit to `LOOP_STATE.md` D-A**: replace the gating clause with the probe's result + path |
| class · tier | **MEASURED (mine, `git ls-files` + reading the probe)** · **CONFIRMED** (the probe itself is five-probe CONFIRMED; the *contradiction* I verified by reading both documents) |

⚠️ This is the **C2 class inverted**: not "absence asserted from one probe", but **"absence-of-a-probe
asserted after five probes had been taken."** A stale gating clause in an *executable* document is worse
than a stale claim in a report, because nothing has to re-read it for the cost to be incurred.

## O-2 🔴 The v1.6 rescue in H1 row 20 is blocked by an HF storage quota, not by pod time

`LOOP_STATE.md` records: **`Sayood/` private HF storage is FULL — 403 "storage limit reached"**, and
lists what it blocks, including *"the older v1.5/v1.6-head … pushes"*. Cross-referenced with registry
§1.4b (*"v1.6's ckpt exists on exactly ONE disk (pod2, currently training) … `Sayood/flagship-v16-ab-ft`
holds NO weights"*), the picture is:

> **The single most at-risk checkpoint in the program cannot be backed up, and the reason is a storage
> quota — a Sayed-side action, not an engineering one.**

It also blocks the v4.1/v4.2/v4.2b ckpt backups that Standing Authorization 1 explicitly instructs an
agent to perform at the next pod2-free moment — so that authorization will fail when it fires.
**Class:** INHERITED (both documents) · **PROVISIONAL** — I could not probe HF from here.
**Cheapest unblock:** the storage upgrade/cleanup already named in `LOOP_STATE.md`; it is listed there as
*"a cheap unblock with broad payoff"* and has not moved.

## O-3 ⚠️ Argoverse 2 fixes ACCESS, not PUBLISHABILITY — do not let H5 row 1 be read as replacing ZOD

Recorded explicitly because the ZOD error (07-26, **C4 — promoted on licence without checking content**)
is one day old and the symmetric error is available here. `LOOP_STATE.md` states it plainly:

> *"**Argoverse 2 = credential-free** … lane connectivity **100 % of 7,692 segments**, 1,052 branch
> points, 6 cities, 20 Hz — **but it is ALSO CC-BY-NC-SA-4.0, so it fixes ACCESS, not PUBLISHABILITY.**"*

So the correct pairing, and both halves are needed:

| corpus | credential-free? | routable lane graph? | commercially usable? |
|---|---|---|---|
| **Argoverse 2** | ✅ (unsigned S3, HTTP 200) | ✅ **byte-verified**, 100 % of 7,692 segments | ❌ **CC-BY-NC-SA-4.0** — NC **and** copyleft |
| **ZOD** | ❌ one access application | ❌ **NONE** — 2-D image-space marking polylines only (retracted 07-26) | ✅ CC-BY-SA-4.0 |
| **Overture `transportation`** | ✅ (2 endpoints, HTTP 200) | ✅ routable, 20 000/20 000 segments ≥2 connectors — but **road-level, not lane-level** | ⚠️ ODbL-1.0; **whether an ODbL-trained MODEL is a Derivative Database is UNSETTLED and not agent-decidable** |

⇒ **H5 row 1's claim is scoped to `Q-TOPO` (research-tier topology substrate) and to nothing else.**
There is still **no corpus that is simultaneously credential-free, lane-graph-bearing and commercially
usable**, and this harvest does not produce one. **Class:** INHERITED · CONFIRMED (three independent
documents agree, and the ZOD half is a logged retraction).

## O-4 (minor, janitorial) Two loose ends in the working tree

| item | state | note |
|---|---|---|
| `4}` in the repo root | **0 bytes, untracked, dated 2026-07-23** | a shell accident (an unquoted brace). Safe to delete; listed so it is not mistaken for a deliverable. |
| `…/incoming/2026-07-26-publishable-corpus-hunt/` | **untracked, empty but for `evidence/`**, mtimes **19:39 and 19:50 today** | ⚠️ **This looks like a LIVE agent's workspace, not stranding.** I did **not** touch it. Flagging for a check *after* that agent reports — deleting or staging a running agent's tree is the hazard the pod2 stand-up correctly refused. |
## O-5 🔴 `CLAUDE.md`'s own illustration of "finish before you start" is a RETRACTED claim

Raised by the H3 sweep, **independently verified by me at both ends**:

| | |
|---|---|
| the binding text | `CLAUDE.md:145-146`, Operating-standard rule **3** — *"Finish before you start… **(LAL-v2 anticipation: implemented, tested, **unmerged 12 days**.**"* |
| the retraction | `RETRACTION_LOG.md:47`, dated **07-21**, class **C4 + C2** — *"'LAL-v2 is implemented but UNMERGED — 12 days idle' … **It merged on 2026-07-09, the day of the intake** (`3784e34`)"* |
| and the residual it left | *"The real residual is **one line** (`taniteval/taniteval/rollout.py:94` keeps 4 of 20 steps)"* — **that is now fixed too**: `rollout.py`'s docstring records *"THE DENSE PATH (added 2026-07-25 — the residual open since 2026-07-09)"*, with `pred_dense`/`gt_dense` wired into 7+ consumers and dedicated tests (MEASURED by the H3 sweep, code-read). |

⇒ **The whole LAL-v2 thread is closed, and `CLAUDE.md` is now the only document still asserting it is
open** — while using it as the *canonical example* of the failure mode it is warning against.

⚠️ **Why this matters more than a stale sentence.** `CLAUDE.md` is loaded into **every agent's context
in this program**. A retracted claim living there is not a documentation defect, it is a **premise
injected into every brief** — precisely the `BOOST_PROGRAM` **C-II / M2** failure ("I inject unverified
premises into agent briefs"), with the highest possible blast radius. The retraction that corrected it
even names the shape: *"a stale escalation demanding work git already contains"*, and the 07-25 C4
lesson adds *"a retraction that does not edit the HEADLINE has not landed."* **This one did not edit
the instruction document.**

🔴 **ESCALATED, NOT EDITED.** I did **not** modify `CLAUDE.md` — changing the binding working-agreements
document is a PI action, not an agent action, and doing it unasked is the wrong precedent regardless of
how obviously correct the edit is. **The fix is one sentence**: replace the LAL-v2 clause with an
example that is still true (the Agent Operating Standard's own table offers `REF-B v2 architecture`,
`the pod ops bundle`, and `the TanitResim maneuver strip`), or re-word it as *"LAL-v2 anticipation was
BELIEVED unmerged for 12 days — it had merged on day one; the belief itself cost the session"*, which
is both true and a **better** illustration, because it is the failure that actually happened.

**Class:** MEASURED (mine — both files read at HEAD) · **CONFIRMED** (raised independently by the H3
sweep and by me; two agents, two paths).


---

# ⭐ TOP 10 ACTIONS, ranked by value ÷ effort — THIS IS WHAT FEEDS THE RECOVERY PLAN

*Ranked across all five inventories plus the live findings. **Effort** is wall-clock on the resource it
actually needs. **Falsifier** is required by `BOOST_PROGRAM` M5.2 — an action that cannot say what would
end it is an activity, not a stream.*

⚠️ **Read the ranking honestly:** actions 1–5 are near-free and near-certain in value; 6–10 are bets
whose value estimate is **PROVISIONAL**. Nothing below is DECISION-GRADE, and nothing below should
decide a GPU-day without the independent reproduction M1 requires.

| # | action | source | closes / prevents | effort | falsifier — what would make this a waste |
|---|---|---|---|---|---|
| **1** | 🔴 **Edit `LOOP_STATE.md` D-A: delete the *"~1-day probe of all 36 PhysicalAI features"* gating clause and replace it with the completed probe's result + path.** | **O-1** | **Prevents an unattended drumbeat from spending ~1 pod-day re-running a probe that is complete at FIVE independent locations** on a fleet where all three pods are busy. The directive is *executable* — nothing has to re-read it for the cost to land. | **one edit, minutes** | If the completed probe is judged not to cover intervention #3's actual need (a *drivable-corridor* channel, not a map). ⇒ then re-scope the clause to that narrower question — but it still must not re-ask "is there a map", which is settled. |
| **2** | 🔴 **`taniteval/taniteval/clhorizon.py:509` — `_data.load_frames` → `_data.load_raw`.** Then run `test_clhorizon.py` and re-emit the affected JSONs. | **H3 S-01** (MEASURED·CONFIRMED at HEAD, found independently by two agents) | The module written **specifically** to un-strand the v4 gate co-primary **raises on its first step** and has evidently never been executed. Until fixed, every re-run of `corridor_departure_rate`@K=185 — the co-primary several BOOST decisions hinge on — must fall back to a one-off `incoming/` driver. **Stream S-1's instrument does not run.** | **1 line**, CPU to fix and unit-test | If `load_raw` turns out not to be surface-compatible. *(Low risk: the committed gate driver already used it, which is why it went unnoticed.)* |
| **3** | ⭐ **Re-emit the 21 `_jack` hierarchy nodes through `paired_episode_cluster_bootstrap`, locally.** | **H1.4** + **H3 D-06** | Three committed artifacts publish `"separated": false` on effects **70×, 47× and 35× their own half-width** — because `_jack`'s `separated` was **one-sided**, so a large *negative* effect renders false by construction. **These files state the opposite of the truth to anyone who greps them instead of reading the report.** | **CPU, minutes — the 27 `windows_*.pt` dumps are on this dev box.** No pod, no GPU. | Nothing. This is a strictly-correcting change with a known-good sibling pattern (`hierarchy.py` was already migrated; only the *artifacts* were not re-emitted). |
| **4** | ⭐ **Re-adjudicate the FIREWALL / negative-control nulls first, before any treatment effect.** 14 identified; 4 at prox > 0.8 (S3 leak probes at n=73, the lead-gate shuffle at n=126, S1's blind-baseline firewall at n=6–20). | **H1.3**, `harvest_index.json → h1.firewall_inversion` | ⚠️ **The inversion nobody stated: for a leak/firewall/shuffle check a null is the DESIRED verdict, so "not separated at n=40" is not a refuted leak — it is a leak we could not see.** The S3 firewall ADMITTED its tasks on nulls projecting to ~2.8× separation. **This is the only action here that can only ever REMOVE results** — and removing a false result is worth more than adding a true one. | **eval-only**, queued behind Bar-A | If the re-scored leaks stay null at higher n ⇒ the firewalls were sound and S3's downstream numbers are clean. **That is a genuinely valuable outcome, not a wasted run** — which is why this ranks above the treatment effects. |
| **5** | ⭐ **Migrate `taniteval/taniteval/planner_p2.py` off `_jack_scalar`/`_jack_paired`** and re-run `analyze_openloop`/`analyze_closedloop` on the existing window dumps. | **H3 S-03** (MEASURED·CONFIRMED — zero `episode_cluster_bootstrap` hits in the file) | `G1_pass`/`G4_pass` — the **CEM-planner-beats-supervised-head** verdicts, and the whole P2 row of the registry — are computed **directly** off the deprecated estimator. Per the program's own measurement it biases the **point estimate** up to **×−4.15 including sign flips**. ⇒ **these verdicts may be reading the wrong sign.** It is the one sibling that missed the migration its three peers got. | **small–medium, CPU-only** (window dumps exist) | If the migrated numbers reproduce the old verdicts ⇒ P2 stands and one more registry row becomes trustworthy. Also a good outcome. |
| **6** | ⭐ **Run the ~1 h read-only AlpaSim `VectorMap` connectivity probe on the free eval pod, and enable `trafficsim` for one scene.** | **H5 row 2** (the *"present but off"* half is MEASURED·CONFIRMED by me: `RUN_RECIPE.md:26` — *"trafficsim (disabled by default)"* — and `alpasim-trafficsim` is a workspace package installed by `uv sync --extra core`) | **We own a map and reactive agents and have switched on neither, for weeks.** The 4-brain program's own scoping says this probe *"gates S1, S2, S4 and HP-4 — four of the nine problems"*, and calls it the **highest-leverage $0 probe in the program**. It attacks `Q-TOPO` and `Q-HP4` at once. | **~1 h, eval pod, read-only, ZERO GPU** | **If the VectorMap carries polygons but no `next_lanes`/`prev_lanes` connectivity** — which is genuinely open, because the prior probe (`gate0_prereq_probe.json`) measured **counts only and its trajectory read ERRORED**. That outcome parks the strategic brain on AlpaSim and routes it to AV2 (action 9) instead. **Both outcomes are informative, which is what makes it worth an hour.** |
| **7** | 🔴 **Correct `CLAUDE.md:145-146` — its example for "finish before you start" is a claim `RETRACTION_LOG.md:47` retracted on 07-21.** | **O-5** (MEASURED·CONFIRMED, both files read at HEAD, raised independently by two agents) | `CLAUDE.md` loads into **every agent's context in this program**. A retracted claim there is not a doc defect — it is **a premise injected into every brief**, i.e. `BOOST_PROGRAM` **C-II/M2** at maximum blast radius. The residual it left (`rollout.py:94`) is now fixed too, so CLAUDE.md is the **only** document still asserting it. | **one sentence** | None. ⚠️ **PI action, not an agent action — I did not edit it.** The honest replacement is *"LAL-v2 was BELIEVED unmerged for 12 days; it had merged on day one, and the belief itself cost the session"* — true, and a better illustration. |
| **8** | **Re-score the closed-loop n≈12 suites at n=40, then 600** — rows 1, 6, 7, 8, 9, 13 of H1.1 (freefloor rung-3 · REF-C planner G1 · refccl tolerance · the imagination FDE leg · E1a's non-junction stratum · DAgger's BC control). | **H1.1 / H1.5 batch 3** | **466 nulls sit at n≈12–24, and this is where the program keeps declaring closure.** The RETRACTION_LOG records **five** consecutive "closed/bound/resolved" verdicts on this direction, each reopened by a cheap follow-up. **This is the sixth cheap follow-up and it is a power check.** Concretely at stake: a **0.25 m peak-XTE** effect and a **0.20 m** imagination effect both read as "no effect" on 12 episodes. | **eval-only**; v1 and REF-C base/XL each have **3 durable copies** | If the first ~6 re-scores at n=40 move nothing ⇒ the n=12 suites were adequately powered after all, and the H1 backlog's closed-loop half is dropped. **Pre-register that stopping rule before starting.** |
| **9** | **Write `stack/scripts/ingest_argoverse2.py`** (mirroring `ingest_nuscenes.py`), so the ~147 MiB AV2 pull has somewhere to land. | **H3 S-05** + **H5 row 1** (both MEASURED·CONFIRMED; I verified driver-absence with `test -f` **and** a call-site grep) | **PI decision #5 in BOOST §5.5 is about to be actioned with nowhere for the result to go.** AV2 is the only credential-free, **byte-verified routable lane graph** we have found (100 % of 7 692 segments), and the adapter + 46 tests already exist. Without a driver the strategic-brain ground truth has **no path from disk into a model**. | **high for this list** — a new driver, CPU-only data engineering | ⚠️ **If action 6 shows AlpaSim's VectorMap already carries connectivity**, AV2 becomes redundant for `Q-TOPO` and this drops several places. **Sequence 6 before 9.** |
| **10** | **Split `BOOST_PROGRAM.md` §3.3's Bar-B sentence into its two clauses and re-source the rate.** | **H5 row 5** (MEASURED·CONFIRMED by me from three raw JSONs) | The *"must fall 2.07×"* gap is **eval-grade**; the *"observed descent −21.6 % per 20k"* is a **TRAINER-LOG** series that reads **1.4900 at step 8–10k while eval reads 2.0739 at 15k** — lower at an earlier step, i.e. **not the same statistic** (C1 + *"a metric NAME is not a metric DEFINITION"*). A rate from one instrument is being applied to a gap from another, inside the recommendation not to restart v4. | **zero** — both eval points already exist | ⛔ **This does NOT claim Bar B is reachable.** Two points fit no rate and the program forbids extrapolation without window + R² + n. If a third eval point shows the eval-grade rate flattening, Bar B is as hard as stated — and we will know it rather than assume it. |

**Runner-up, listed because it is a PI action with broad payoff and it blocks a rescue:**
**O-2 — the `Sayood/` HF storage 403.** It blocks backing up **v1.6**, the single most at-risk checkpoint
in the program (registry §1.4b: *one disk, HF repo holds no weights*), **and** it will make Standing
Authorization 1's ckpt-backup instruction fail when it fires. `LOOP_STATE.md` already calls the cleanup
*"a cheap unblock with broad payoff"* and it has not moved.

---

## What this harvest adds to `BOOST_PROGRAM` §4 — the two missing streams

M5 was **overruled by the PI: *"no concentration on only three stream, at least five streams"***, and §4
still names only three (S-1 closed-loop measurability · S-2 the selector · S-3 the v2-corpus arm). The
harvest supplies the missing two — **and both are eval-only**, so they add breadth **without adding GPU
contention**, which was the specific cost M5 was raised to avoid. Each carries its own falsifier, as
M5.2 requires.

| stream | content | resource | falsifier (pre-registered before it starts) |
|---|---|---|---|
| **S-4 — RE-ADJUDICATION** | Clear the H1 backlog in the H1.5 priority order: **firewalls first** (action 4), then the `_jack` re-emissions (action 3) and `planner_p2` (action 5) which are **local and free**, then the closed-loop suites (action 8), then v1-vs-REF-C-XL at n=600. | **eval pod only, no training.** The 600-ep protocol is **already built and proven** (`RESULT_v1_600ep.json`). | **If the first 20 re-scores flip fewer than 2 verdicts**, the ×3.4 shrinkage does not survive contact with moving point estimates ⇒ stop, and record that the 789 nulls are mostly real. |
| **S-5 — TOPOLOGY SUBSTRATE** | Actions 6 → 9: probe AlpaSim's VectorMap connectivity and switch on `trafficsim`; if connectivity is absent, write the AV2 ingest driver and take the 147 MiB pull. Feeds `Q-TOPO`, `Q-HP4`, S1/S2/S4. | ~1 h eval pod, then CPU data engineering. **Zero GPU.** | **If AlpaSim has no lane connectivity AND the AV2 pull is refused**, `Q-TOPO` stays blocked ⇒ the strategic brain is **parked with its state banked**, explicitly, rather than drifting. |

⚠️ **One caveat on S-5 that must travel with it (O-3):** AV2 is **`CC-BY-NC-SA-4.0`** — non-commercial
**and** copyleft. It fixes **ACCESS, not PUBLISHABILITY**. There is still **no corpus that is
simultaneously credential-free, lane-graph-bearing and commercially usable**, and this harvest does not
produce one. Do not let S-5 be read as replacing the ZOD line — ZOD is commercially usable and has **no
lane graph at all** (retracted 07-26).
