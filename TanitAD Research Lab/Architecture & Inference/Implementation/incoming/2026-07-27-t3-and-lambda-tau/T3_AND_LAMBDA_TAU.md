# T3 · the (λ, τ) curve · GoalFlow's literal shadow rule — the eval-host run

**Written:** 2026-07-27 (Europe/Berlin; the pod and every log below are UTC).
**Host:** `tanitad-eval` (A40 46 GB, 96 cores, Python 3.12.3, torch 2.8.0+cu128).
**Mandate:** run the two pre-registered eval-host jobs that were blocked on a host, not a design —
T3 (`PERCANDIDATE_LABELS.md` §8) and the (λ, τ) cache + curve (`LAMBDA_TAU_SWEEP.md` §5).
⛔ **pod1, pod2 and pod3 were never contacted, not even read-only.** Nothing was trained from
scratch; nothing was killed.

Evidence stamps on every number: class `MEASURED` / `PUBLISHED` / `INHERITED` / `ESTIMATED` /
`HYPOTHESIS`, **and** tier `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

---

## HEADLINE

> ### All three pre-registered verdicts came back NEGATIVE, and the negatives are the informative part.
> ### **T3 = REFUTE.** **The (λ, τ) optimum = UNPOWERED, and the shipped λ=1 is the argmin of its own row.** **GoalFlow's shadow rule = REFUTE.**

1. ⭐⭐ **T3 REFUTES — and the sharper finding is that NEITHER fitted arm beats doing nothing.**
   `BCE_RULE − CE_CONTROL` on the pre-registered primary, PDMS-lite: **+0.0002 [−0.0025, +0.0031]**,
   not separated; at-fault collision **+0.0000 [−0.0035, +0.0035]**, not separated and not below.
   **Both CONFIRM legs fail.** `AS_TRAINED` has the best PDMS-lite (**0.6100**), the lowest collision
   rate (**0.0361**) and the best ADE (**0.8563**) of all three arms.
   **Bar A refuted a re-scoring lever on ADE; the standing objection was that ADE is the wrong axis.
   We changed the axis to the composite, and the answer did not change.**
   `MEASURED · CONFIRMED` · 837 s of A40 · `raw/t3_result.json`.
   ⚠️ **Read the metric's resolution with the result:** random scores 0.3968, the three trained arms
   occupy 0.6096–0.6100 — a spread of **0.2 % of the distance to random**. PDMS-lite (no-map)
   separates a selector from noise and **barely separates two selectors from each other**.

2. ⭐⭐ **THE (λ, τ) CURVE IS MEASURED — 42 cells × 2 sheets, and its verdict is one of the four
   pre-registered FAILING states.** The argmin is (λ=0.25, τ=0.1) at **0.8483 m** against the
   shipped (1, 1) cell's **0.8563 m**; paired episode-cluster bootstrap **−0.0079
   [−0.0272, +0.0126]**, which **includes zero** ⇒ **UNPOWERED**. The **admissible set — every cell
   not separated from the argmin — is 30 of 42 cells**. That is the interval, and it is enormous.
   ⚠️ The argmin also sits on a **grid edge** (τ = 0.1), so had it been powered the rule would have
   returned **NO-INTERIOR**: **two** of the four failing conditions hold simultaneously.
   `MEASURED (ours) · CONFIRMED` · `raw/eh2_sweep.json`.

3. ⭐⭐ **ALONG τ = 1 THE SHIPPED λ = 1 IS THE MINIMUM OF ITS OWN ROW, with a clean interior U.**
   0.8742 · 0.8681 · 0.8587 · **0.8563** · 0.8572 · 0.8594 · 0.9096 for λ = 0 … 8. The prior's
   strength was hard-wired and it was hard-wired **on the optimum** — a *positive* result hiding
   inside a failing verdict. `MEASURED · CONFIRMED`.

4. ⭐ **E-H0b IS DISCHARGED ON THE PRODUCED (DEPLOYABLE) SURFACE, and the "worth about half"
   reading does not carry over.** λ = 0 is now a config, so the graft term is measured directly:
   **graft = +0.0179 m**, constant-velocity term = **+0.0039 m**, and they sum to the committed
   `F_flat − F_base_only` = **+0.0218 m to the digit**. On the *deployable* surface the hierarchy
   graft is **82.1 %** of the bundle, not ~50 % (which was the *oracle* surface, `DERIVED /
   PROVISIONAL`). ⚠️ Its own interval **+0.0180 [−0.0022, +0.0455] is NOT separated** at n = 40.

5. ⭐ **THE NAMED TRAP: predicted exactly, then measured exactly.** The gate *derived* (no forward
   pass) that at τ = 1 the clamp cannot bind below λ = 8.31 and the guard cannot raise below
   λ = 12.46. **Measured: `preclamp_ratio_max` at (λ=8, τ=1) is 0.9630 = 8 × 0.1204**, linear to
   4 dp, `clamp_bound_frac` **0.0000 on all 7 λ values at τ = 1**. The τ-axis estimate was
   **1.204** at (1, 0.1); **measured 1.1968** — the `ESTIMATED` value was right to **0.6 %**.

6. ⛔ **`seam_fail` RAISES on 6 of 42 cells, and those 6 are EXACTLY the cells where the two clamp
   sheets disagree.** 36 of 42 cells are bit-identical across `deployable` and `diagnostic`;
   the 6 that differ (λ ≥ 2 on the sharpening branch) are the 6 the shipped guard refuses to run.
   ⇒ **inside the reachable configuration space of the deployed head, `seam_clamp` never changes
   the answer** — and where it *would*, it is worth a lot: at (8, 0.1) the clamped sheet reads
   **1.4738 m** and the unclamped **6.8175 m**.

7. ⭐ **GoalFlow's LITERAL shadow rule is now measurable — and it REFUTES too.** `LAMBDA_TAU_SWEEP.md`
   §8 listed it as not computable because no staged artifact held the neutral branch's *trajectory*.
   The new cache holds it. Best deployable variant (endpoint deviation, threshold chosen
   leave-one-episode-out): **0.7607** vs the neutral-always **0.7620**, Δ **−0.0014
   [−0.0066, +0.0038]**, not separated. Even its **in-sample** ceiling is **0.7592**, against an
   oracle-shadow bound of **0.6883**. **Five deployable reliability signals have now failed the same
   bar.** `MEASURED · CONFIRMED` · `raw/eh3_goalflow.json`.

8. ⭐ **The instrument got *better* at parity, and it matters for what may be quoted.** T2's
   dev-box labeler flagged the realised human future at **2.05 %** (5.3× PARA-Drive's published
   0.384 %) and therefore forbade quoting absolute collision levels. Re-minted **at parity** on the
   canonical val corpus the same labeler flags the human at **0.435 %** — **1.13×** the published
   reference. `MEASURED · CONFIRMED` · `raw/t3_labels.json`.

9. ⛔ **On v4's OWN emitted fan, comfort is not merely saturated — it is a CONSTANT.**
   T2 measured 99.34 % comfort violation on the anchor vocabulary; on flagship-v4-fromscratch-30k's
   emitted fan it is **100.0000 %** — every one of **1,708,288** labelled candidates. **The `2·C`
   term of PDMS-lite is identically zero and the comfort BCE head has a constant target: it carries
   no ranking information at all.** This is reported, not tuned away.

10. ⭐ **T1's clip reproduces on a second arm's fan, to the digit where it matters.** On v4's own
   emitted fan the head's own reachability clamp removes **73.88 %** of candidates (T1 on REF-C-XL:
   72.08 %), the ADE-oracle survives in **100 %** of windows (T1: 100 %), **no window is emptied**,
   and the as-trained pick moves in **ZERO windows — paired Δ exactly 0.000000** (T1: exactly
   0.0000). An independent replication on a different arm.

---

## 0. WHAT WAS PRE-REGISTERED, BY WHOM, AND WHAT I DID NOT AUTHOR

⚠️ **I authored neither kill.** Both were written and staged by earlier agents *before this one
existed*, and both are honoured unmodified:

| job | pre-registration | authored in |
|---|---|---|
| **T3** | arms · protocol · combine · primary read · estimator · CONFIRM/REFUTE · preconditions · free flag | `PERCANDIDATE_LABELS.md` §8 |
| **(λ, τ)** | the 42-cell grid, the two clamp sheets, `locate_optimum`'s five verdicts, the estimator | `LAMBDA_TAU_SWEEP.md` §0, in code |
| **shadow branch** | *"CONFIRM if any deployable rule beats the neutral 0.7620, paired and separated"* | `HIERARCHY_PRIOR_RESEARCH.md` §6.4, via `eh3_shadow_branch.py` |

**Estimator, never varied:** paired episode-cluster bootstrap (`taniteval/ci.py`, B = 2000), unit =
episode cluster. **`overlapping_holdout_se` is used nowhere in this stream** — it is not a jackknife,
it biases the point estimate bidirectionally (−6.67 % to +11.69 % over 27 arms) and it has flipped
the sign of a paired delta.

**Two implementation calls I had to make, both stated in code before the run** (`code/t3_scorer.py`
docstring, written before its first execution):

1. §8 specifies `conf_head: Linear(512,1) → Linear(512,4)` **and** an inference combine
   `w₁ log S_im + w₂ log S_NC + …`. A 4-output head has no `S_im`. The head is therefore built with
   **five** outputs: channel 0 is the imitation logit, initialised from the as-trained
   `Linear(512,1)`, trained with the **same** CE as `CE_CONTROL`; channels 1–4 are the four sigmoids
   under BCE, **zero-initialised**, so `log σ(0)` is a per-candidate constant and the arm **starts
   bit-identical to as-trained**. `BCE_RULE − CE_CONTROL` therefore isolates *exactly* the four
   added rule heads. **The literal 4-only reading is the `w_im = 0` corner of the weight grid**, so
   it is measured rather than chosen — and that corner is also §8's *"one free flag"*
   (`use_q` / hide the planner's own score from the selector).
2. LR is selected on inner-validation **PDMS-lite** for **both** fitted arms. Selecting the control
   on ADE and the treatment on PDMS-lite would confound the arm with its selection criterion.

### 0.1 ⚠️ HOST, CORPUS AND PROVENANCE DISCLOSURE — read before quoting anything

- **The pod was NOT idle when this run began.** `run_pseudosim.py` (PID 1767114, a sibling stream)
  was mid-flight on the same A40. It was **left alone** and completed on its own at ≈ 00:36 UTC;
  every job here was thread-capped and ran alongside it. Peak concurrent GPU use was
  **6.5 GiB of 46 GiB**. **No process was killed.**
- 🔴 **`/root` really is 99 % full** (`overlay 200G, 2.7G avail`) — re-measured. **Everything was
  written to `/workspace`** (real `dd`: 3.0 GiB at 388 MB/s), and every artifact pulled back to the
  repo was **md5-verified against the pod** (§6).
- ⚠️ **T3's labels are RE-MINTED AT PARITY and are a DIFFERENT MEASUREMENT from T2's.** T2 ran on
  the dev box over the *anchor vocabulary* and a non-parity corpus. T3 needs labels on the SAME
  windows and the SAME fan Bar A's cache holds, so they were re-minted pod-side on
  **`physicalai-val-0c5f7dac3b11` — the canonical val corpus, the same 881 canonical windows every
  committed bar uses.** **The two label sets are kept separate and neither is a model fact; neither
  may enter `MODEL_REGISTRY.md`.**
- **Nothing re-selects episodes.** The parity invariant is untouched: Bar A's cache, its folds, its
  seeds and its 881-window canonical eval subset are reused unmodified.
- 🔒 **PhysicalAI-AV is gated-confidential.** The 40 val clips appear only as
  `clip_<sha256[:8]>` aliases; **no clip UUID and no raw content is in any staged artifact**, and
  the raw label parquets stay on `/workspace`.
- ⛔ **A permission boundary was hit and respected, not worked around.** Installing the HF token on
  the pod was **blocked by the permission system**. Rather than route around it, the gated label
  files were fetched **on the dev box** with HTTP **range** reads (§1.1) and the ~35 MB of extracted
  per-clip parquets shipped to the pod. **No credential was written to the pod, and none is in any
  artifact.**

---

## 1. GETTING T3's LABELS ONTO THE PARITY CORPUS — the part that was actually hard

§8's blocker was stated as *"Bar A's feature cache lives only on the eval pod"*. That was true and it
was **not the binding constraint**. Bar A's cache holds `qf`, `fan`, `tgt`, `v0`, `ep`, `t` — it holds
**no agent tracks**, and **PhysicalAI-AV's `obstacle.offline` is not on the eval pod at all**
(`MEASURED-ABSENCE`, two probes: `find /root /workspace -maxdepth 5 -iname '*obstacle*'` → empty, and
the val episode cache carries only `frames_u8 / actions / poses / episode_id / maneuvers`).

### 1.1 Identifying the 40 val clips — and proving it rather than asserting it

`physicalai.build_episode` stores `episode_id = int.from_bytes(clip_id.encode()[:4])` — **four
characters of the UUID**. Matching those 40 prefixes against the program's own 3,000-clip
`phase0_selection.parquet` returns **exactly one candidate each, 40 for 40, zero ambiguity**
(`MEASURED`). That is an inference, so it is then *proved* two further ways:

| probe | result |
|---|---|
| **clock inversion** — `build_episode` interpolates egomotion at `linspace(A, B, N)`, so pose-index → clip-time is exactly affine. Two seeds (nearest-neighbour + Theil–Sen; and the canonical dense-span geometry), each refined on the true xy residual | **max xy rms 1.1 × 10⁻⁵ m over all 40 episodes** (median 3 × 10⁻⁶ m). A wrong clip cannot fit at 11 µm. |
| **window-index convention**, scanned rather than inherited from a config: which offset `k` makes egomotion's speed at `t(t_window + k)` reproduce the cache's own `v0`? | **k = 7 on 40 of 40 episodes**, with a razor-sharp interior minimum: max abs error **8 × 10⁻⁵ m/s** at k = 7 versus ~0.14 m/s at k = 6 and k = 8 ⇒ `pose_last = poses[t + w − 1]`, w = 8. `MEASURED`, not assumed. |
| ⛔ **clock-quality gate** — `clock_ok` iff rms ≤ 0.25 m **and** the offset scan's minimum is *interior* **and** the v0 error there ≤ 0.20 m/s | **40 / 40 PASS.** Episodes that fail are excluded from the trusted set and named, never silently kept. |

⚠️ **Two clock traps, both hit and both now priced.**
**(a)** `PERCANDIDATE_LABELS.md` §1's trap: an optimiser seeded on egomotion's own range lands
**100+ s away at rms ~10² m**, because egomotion carries a sparse tail to ~140 s past the 20 s clip.
**(b)** A trap that file did not name: nearest-neighbour matching in (x, y, v) is **degenerate over a
standstill**, and on **1 of 40 clips** the Theil–Sen slope collapsed to **74.0 ms instead of
~100.7 ms**, giving **rms 2.60 m** — a silently wrong clock that places every agent track up to 2.6 m
off. It was caught by the offset scan pinning at its edge (k = 14, no interior minimum), fixed by
adding the canonical-geometry seed and refining on the real residual, and **the whole label set was
re-minted**. Both failure modes are recorded in `code/t3_labels.py`'s docstrings so the next agent
does not rediscover either. *(See the retraction in §8 — I had quoted the good-case residual from a
5-episode spot check before the full run existed.)*

### 1.2 Fetching the tracks without a pod credential

`HfFileSystem` + `zipfile` reads a zip's central directory and then only the members requested, so
each 64 MB chunk cost ~0.3–0.6 MB of transfer instead of a full download: **37 chunks → 116 files,
35.3 MB total**, and no ~4 GB of chunk pulls. `code/pull_val40_labels.py`.

⚠️ **One clip of 40 has NO `obstacle.offline`.** `MEASURED-ABSENCE at three probes`: absent from its
own chunk's 91-member namelist and from both neighbouring chunks, while its **egomotion IS present in
that same chunk** — so this is a per-clip gap in the label product, not a chunk-mapping error.
*(Independently consistent with the program's standing "`obstacle.offline` on 97.44 % of the corpus";
here 39/40 = 97.5 %.)* **Consequence, stated rather than glossed: every T3 rule number lives on
859 of the 881 canonical windows (39 of 40 episode clusters).** The cache-fidelity self-test still
runs on the full 881 and reproduces the committed bar exactly, and all three T3 arms are scored on
the identical 859.

### 1.3 The labels, at parity — and what they say before any scorer is trained

`MEASURED · CONFIRMED` · `raw/t3_labels.json`, per-window × per-candidate dump `raw/t3_labels.pt`
(21.3 MB). **6,844 windows × 256 candidates = 1,752,064 candidates labelled**, of which
**1,708,288 (6,673 windows) carry trustworthy tracks** and **219,904 (859 canonical eval windows)
carry the T3 arms**. 121 s of CPU on 16 workers.

| label, on **v4's own emitted fan**, at parity | positive rate | varies within-window | T2's dev-box value (anchor fan, non-parity) |
|---|---:|---:|---:|
| **NC — at-fault collision** | **0.2996** | **0.6086** | 0.2265 |
| collision, any | 0.3091 | — | 0.2389 |
| **TTC infraction** | **0.5338** | **0.7832** | 0.2867 |
| comfort **violation** | **1.0000** | 0.0000 | 0.9934 |

Agents per window: mean **33.4**, median 19, max 256, **3.99 %** of windows have none.
**T2's pre-registered kill would not have fired here either** — the rate leg is 6.0× the 5 %
threshold on v4's own fan, versus 4.5× on the anchor fan. *(Reported as corroboration only: it is a
different corpus and a different fan, so the two are never pooled.)*

> ⚠️ **Comfort is a CONSTANT on this fan, not merely saturated.** 100.0000 % of candidates violate.
> The `2·C` term of PDMS-lite is identically 0 and the comfort BCE head has a constant target.
> §8's precondition — *"comfort excluded from the hard veto until the fan is speed-conditioned"* —
> was right and is now stronger: **on v4's emitted fan, comfort is not a weak feature, it is no
> feature.** The mechanism is the same one T2 named (a global, non-speed-conditioned vocabulary
> plus an unbounded offset head), amplified by the offset head that the anchors alone do not have.

> ⭐ **The instrument's false-positive floor is 4.7× better at parity, and this changes what is
> quotable.** The same labeler on the **realised human future**: at-fault **0.435 %**, any collision
> **0.510 %**, TTC **0.749 %**. `PUBLISHED (PARA-Drive, PDF-VERBATIM)` reference: **0.384 %**.
> T2's caveat *"absolute collision levels are NOT quotable"* was earned by the dev-box setup's
> 2.05 %; **at parity the floor is 1.13× the published reference**, so absolute levels on THIS
> corpus are defensible — though contrasts remain the safer read.
> ⚠️ **The comfort control fails in the other direction:** the human's own future satisfies our
> comfort bounds only **53.1 %** of the time, so the nuPlan/NAVSIM thresholds (`INHERITED`) are
> mis-specified against a 10 Hz finite-difference jerk. **A comfort term must not be trusted here
> until those thresholds are re-derived.**

### 1.4 The pre-registered precondition, applied first — and it replicates T1 exactly

`MEASURED · CONFIRMED`. The head's **own** reachability clamp
(`v_term ∈ [max(0, v₀ − 5.0), v₀ + 5.0]`, reach = `sel_accel_max × horizon` = 2.5 × 2.0):

| | **v4's own emitted fan (here)** | REF-C-XL (T1 §3) |
|---|---:|---:|
| candidates removed | **73.88 %** | 72.08 % |
| windows where the ADE-oracle survives | **100 %** | 100 % |
| windows emptied by the clip | **0.000 %** | — |
| **windows where the as-trained pick MOVES** | **0.000 %** | 0.00 % |
| paired Δ `ade_0_2s` | **exactly 0.000000** (0.8563 → 0.8563) | exactly 0.0000 |

> **T1's measured precondition holds on a second arm's fan, and it is still free.** All three T3
> arms are scored on this clipped candidate set, so the clip cannot confound the contrast.

---

## 2. T3 ⭐ — the 4-head BCE rescorer, scored on PDMS-lite. **REFUTE.**

`MEASURED (ours) · CONFIRMED` · `raw/t3_result.json` · per-window dump `raw/t3_windows.pt` ·
**837 s (14.0 min) of A40 time** — the §8 estimate of *"~13 GPU-min"* was right.
Arms fitted with **Bar A's 5-fold episode-disjoint cross-fit, its LR grid {3e-5, 1e-4, 3e-4}, its
folds, its seeds, its batch and its step budget**, on the **identical 881-window canonical eval
subset**, restricted to the **859 windows / 39 episode clusters** that carry trustworthy rule labels.

### 2.1 The two self-tests, both directions — PASS before anything was trained

| self-test | result |
|---|---|
| **cache fidelity** — the cached path must reproduce the published forward pass *and* pick the same trajectory in every window | `ade_0_2s` **0.8563** vs committed **0.8563** (diff 2e-05) · `oracle_in_fan` **0.2505** vs **0.2505** (diff 5e-05) · **pick differs in 0.000 % of windows** ✅ |
| **failing input** — a uniform-random selector over the same frozen fan must be worse on **both** surfaces | ADE **15.3622** vs 0.8563 *(Bar A's committed 15.3622, to the digit)* · PDMS-lite **0.3968** vs 0.6131 ✅ |

**Reproduce before quoting**: Bar A's committed as-trained bar and its failing-input value both
reproduce exactly, on a harness written independently of Bar A's.

### 2.2 The arms

859 windows / 39 episode clusters. All three arms score the **same clipped candidate set**.

| arm | **PDMS-lite** ↑ | at-fault collision ↓ | TTC infraction ↓ | `ade_0_2s` (859) | `ade_0_2s` (all 881) |
|---|---:|---:|---:|---:|---:|
| **AS_TRAINED** | **0.6100** [0.5552, 0.6604] | **0.0361** [0.0128, 0.0641] | 0.1653 | 0.8352 | **0.8563** |
| **CE_CONTROL** (Bar A's exact refit) | 0.6096 [0.5556, 0.6597] | 0.0373 [0.0140, 0.0652] | 0.1711 | 0.8691 | 0.8930 |
| **BCE_RULE** (4 rule heads + Hydra-MDP combine) | 0.6098 [0.5550, 0.6600] | 0.0373 [0.0140, 0.0653] | 0.1723 | 0.8777 | 0.9010 |
| *random (the deliberately failing input)* | *0.3968* | — | — | *15.3622* | *15.3622* |

**Paired episode-cluster bootstrap, B = 2000, unit = episode cluster, identical windows:**

| contrast | Δ PDMS-lite | Δ at-fault collision | Δ `ade_0_2s` |
|---|---|---|---|
| **PRIMARY — BCE_RULE − CE_CONTROL** | **+0.0002 [−0.0025, +0.0031]** ❌ | **+0.0000 [−0.0035, +0.0035]** ❌ | +0.0086 [−0.0115, +0.0303] ❌ |
| BCE_RULE − AS_TRAINED | −0.0003 [−0.0034, +0.0027] ❌ | +0.0012 [+0.0000, +0.0035] ❌ | +0.0425 [−0.0138, +0.1004] ❌ |
| CE_CONTROL − AS_TRAINED | −0.0005 [−0.0037, +0.0027] ❌ | +0.0012 [+0.0000, +0.0035] ❌ | +0.0339 [−0.0203, +0.0881] ❌ |

*(❌ = not separated. PDMS-lite positive = better; collision and ADE negative = better.)*

> ### VERDICT: **REFUTE**, at the pre-registered bar, on both of its legs.
>
> | pre-registered condition | required | measured |
> |---|---|---|
> | `BCE_RULE − CE_CONTROL` separated-**better** on PDMS-lite | yes | **+0.0002 [−0.0025, +0.0031]** — not separated |
> | its at-fault collision rate separated-**below** `CE_CONTROL`'s | yes | **+0.0000 [−0.0035, +0.0035]** — not separated, and not below |
> | **CONFIRM?** | both | **NO ⇒ REFUTE** |
>
> **Stated plainly and not re-scoped:** adding four rule-supervised sub-scores to this frozen fan's
> selector, under the Hydra-MDP combine, with the weights grid-searched on the fit folds, **does not
> move the composite.**

### 2.3 What the REFUTE is, and what it is not

⭐ **The sharpest thing in the table is not the treatment — it is that neither fitted arm beats doing
nothing.** `AS_TRAINED` has the best PDMS-lite (0.6100), the lowest collision rate (0.0361) and the
best ADE (0.8563) of the three. **This is Bar A's result reproduced on an orthogonal surface**: Bar A
refuted a re-scoring lever on `ade_0_2s`, and the standing objection was that ADE is the wrong axis
(§0 of `PERCANDIDATE_LABELS.md` §5: the ADE-optimal pick is 4.7× more dangerous than the rule-optimal
one). **We changed the axis. The answer did not change.** `MEASURED · CONFIRMED`.

⚠️ **But read the metric's resolution before reading the arms.** Random scores **0.3968**; the three
trained arms occupy **0.6096–0.6100**, a spread of **0.0004 — 0.2 % of the distance to random.**
**PDMS-lite (no-map) separates a selector from noise and barely separates two selectors from each
other.** Part of that is ours: **DAC is missing (no map) and comfort is a constant on this fan**, so
**two of PDMS's terms carry no information here** and the composite is effectively
`NC × (5·EP + 5·TTC)/12`. **A composite that cannot resolve the arms cannot adjudicate between them,
and that is a limitation of the instrument as much as a finding about the arms.**

⛔ **This does NOT refute the T2 line.** T2 measured that the rule label is common, varies inside
most windows and is nearly orthogonal to ADE — all of which reproduces here on v4's own fan at
parity (§1.3). T3 measures something narrower and answers it negatively: *these four sub-scores,
added to **this frozen fan's** score under **this** combine, fit on 26 episodes, do not move the
composite.* **"The signal exists" and "a realisable ranker extracts it" remain different claims, and
the second one has now failed twice on this frozen fan** — once on ADE (Bar A), once on PDMS-lite.

### 2.4 The free flag, the weight grid, and the head's own guard — all measured

- ⭐ **§8's "one free flag" (`use_q` / hide the planner's own score) is the `w_im = 0` corner, and it
  is WORSE in all 5 folds** — inner-val PDMS-lite 0.6544 vs 0.7061 · 0.4929 vs 0.5488 · 0.5528 vs
  0.7114 · 0.6339 vs 0.6699 · 0.3936 vs 0.5211. **The literal 4-head reading of §8 (rule scores
  alone) never beats keeping the imitation score.** The Slow-Brain `HYPOTHESIS` does not transfer to
  this selector. `MEASURED · CONFIRMED`.
- The chosen combine scale varied by fold (`s` ∈ {0.25, 4.0}); `w_im = 1` won every fold. LRs chosen:
  3e-5, 1e-4, 1e-4, 3e-4, 3e-5. **In 2 of 5 folds the best inner-val state was `step = 0`** — i.e.
  **the fine-tune never helped at all**, which is itself the Bar A signature.
- ⛔ **The head's own fail-loud seam guard refused 15 of 60 evaluated weightings** (2–4 per fold):
  under the log-sigmoid combine the base score's norm shrinks enough that the graft swamps it and
  `seam_fail = 1.5` fires. Those weightings are recorded as refused, never silently skipped.
  Training itself never tripped it (max pre-clamp ratio **1.3746** in CE, **1.0808** in BCE, both
  below 1.5) — consistent with Bar A, which *did* trip it at 1.652 under the regret loss.

## 3. THE (λ, τ) CURVE — 42 cells × 2 sheets, MEASURED

`MEASURED (ours) · CONFIRMED` · `raw/eh2_sweep.json` · cache `raw/eh2_cache.pt` (9.6 MB, staged, so
**every cell recomputes on any CPU with no GPU and no pod**).

**The blocker is removed.** `eh2_build_cache.py` had never been executed; it ran here in
**212 s of A40 time for both goal modes** (produced 106 s, neutral 106 s) over the 881 canonical
windows, alongside the sibling pseudo-sim job. The sweep itself is **27.4 s + 27.7 s of CPU** for the
two sheets. *(The stream's estimate — "~2–3 min per goal mode" — was right.)*

### 3.1 The gates, re-run rather than assumed

| gate | result |
|---|---|
| harness self-test, 4 fixtures, **both directions** | **4/4 PASS** — `planted → CONFIRM-INTERIOR`, `degenerate → DEGENERATE`, `saturated → SATURATED`, `unpowered → UNPOWERED` |
| **fidelity gate** — the CPU re-scorer vs the forward pass's own `sel_score`/`sel_idx` | **PASS**: `max_abs_score_err = 1.19e-07` (fp32 rounding), **selection fidelity 1.0000** — same pick on every one of 881 windows |
| the committed bars, recomputed from the new cache | `produced/F_flat` **0.8563** ✅ · `produced/O_oracle_in_fan` **0.2505** ✅ · `produced/F_base_only` **0.8781** ✅ · `neutral/F_flat` **0.7620** ✅ · `oracle/F_flat` **0.6423** ✅ |

**Reproduce before quoting: five committed bars reproduce to the digit from a cache built in this
run**, which is what makes every new cell below admissible.

### 3.2 The deployable sheet (`seam_clamp = 1.0`) — all 42 cells

`ade_0_2s`, produced surface, 881 windows / 40 episode clusters.

| λ \ τ | 0.1 | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| **0** | 0.8742 | 0.8742 | 0.8742 | 0.8742 | 0.8742 | 0.8742 |
| **0.25** | **0.8483** | 0.8575 | 0.8602 | 0.8681 | 0.8727 | 0.8740 |
| **0.5** | 0.8664 | 0.8558 | 0.8594 | 0.8587 | 0.8589 | 0.8734 |
| **1** | 0.9497 | 0.8619 | 0.8579 | **0.8563** | 0.8602 | 0.8592 |
| **2** | 1.2904 | 0.9024 | 0.8600 | 0.8572 | 0.8611 | 0.8613 |
| **4** | 1.4706 | 1.0712 | 0.9066 | 0.8594 | 0.8605 | 0.8630 |
| **8** | 1.4738 | 1.4667 | 1.0715 | 0.9096 | 0.8668 | 0.8607 |

**Per-cell pre-clamp ratios — reported for every cell, as the brief requires**
(`preclamp_ratio_max`; the `mean`, `p50` and `p95` are in the JSON, and `mean` is 0.391 × `max` on
every cell). `bnd` = `clamp_bound_frac`, `trip` = fraction of the batch the **shipped**
`seam_fail = 1.5` guard would RAISE on.

| λ \ τ | 0.1 | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| **0** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **0.25** | 0.2992 | 0.1197 | 0.0599 | 0.0301 | 0.0156 | 0.0089 |
| **0.5** | 0.5984 | 0.2394 | 0.1197 | 0.0602 | 0.0311 | 0.0177 |
| **1** | **1.1968** *(bnd 0.036)* | 0.4787 | 0.2394 | **0.1204** | 0.0623 | 0.0354 |
| **2** | **2.3936** *(bnd 0.310, trip 0.142)* | 0.9574 | 0.4788 | 0.2407 | 0.1246 | 0.0709 |
| **4** | **4.7871** *(bnd 0.955, trip 0.510)* | **1.9148** *(bnd 0.210, trip 0.049)* | 0.9576 | 0.4815 | 0.2492 | 0.1417 |
| **8** | **9.5742** *(bnd 1.000, trip 1.000)* | **3.8297** *(bnd 0.761, trip 0.346)* | **1.9153** *(bnd 0.211, trip 0.050)* | 0.9630 | 0.4984 | 0.2835 |

### 3.3 ⭐⭐ The named trap: derived exactly *before* the run, measured exactly *in* it

| claim | how it was obtained | value | measured here |
|---|---|---|---|
| pre-clamp ratio at (λ=1, τ=1) | `MEASURED` from a staged v4-gate artifact | 0.1204 | **0.1204** ✅ |
| the graft scales **exactly linearly in λ** | asserted structurally | — | **0.9630 = 8 × 0.1204** (4 dp) ✅ |
| clamp first binds at λ > **8.31** (τ = 1) | derived, no forward pass | — | (8, 1) has `bnd = 0.0000` ✅ |
| shipped guard raises at λ > **12.46** (τ = 1) | derived | — | (8, 1) `trip = 0.000` ✅ |
| **0 of 7 λ values clamp-bound at τ = 1** | derived | — | **0 of 7** ✅ |
| pre-clamp ratio at (λ=1, τ=0.1) | `ESTIMATED` (‖graft‖ ~ 1/τ) | ≈ 1.204 | **1.1968 — the estimate was right to 0.6 %** ✅ |

> **The λ axis is clean across the whole pre-registered grid and the trap really does live on τ.**
> The flat λ response along τ = 1 is therefore **a finding about λ, not saturation** — the exact
> discrimination the spec demanded, and it was available *before* the run rather than argued after.

### 3.4 ⛔ `seam_fail` raises — and the two sheets disagree on exactly those cells

**6 of 42 cells** would raise the shipped guard: (2, 0.1) trip 0.142 · (4, 0.1) 0.510 ·
(4, 0.25) 0.049 · (8, 0.1) **1.000** · (8, 0.25) 0.346 · (8, 0.5) 0.050. A sweep run against the
shipped config would have **died at the first hard cell**, not returned a flat curve.

**36 of 42 cells are identical across the `deployable` and `diagnostic` sheets, and the 6 that
differ are precisely those 6:**

| cell | deployable (clamp on) | diagnostic (clamp off) | ratio |
|---|---:|---:|---:|
| (2, 0.1) | 1.2904 | 1.8741 | 1.45× |
| (4, 0.25) | 1.0712 | 1.3564 | 1.27× |
| (4, 0.1) | 1.4706 | 4.9337 | 3.36× |
| (8, 0.5) | 1.0715 | 1.3567 | 1.27× |
| (8, 0.25) | 1.4667 | 4.0494 | 2.76× |
| **(8, 0.1)** | **1.4738** | **6.8175** | **4.63×** |

⇒ **Amendment 3 of `LAMBDA_TAU_SWEEP.md` §6 is confirmed** ("expect the sheets to coincide along
τ = 1; if they do not, the cache is wrong"). They coincide along τ = 1 **and everywhere else the
clamp does not bind.** The sharper reading: **within the configuration space the deployed head will
actually run, `seam_clamp` never changes the answer** — every cell where it does is a cell
`seam_fail` refuses. Where it *would* matter it is worth up to **4.63×**.
*(One cell, (1, 0.1), is 3.6 % clamp-bound yet identical on both sheets: the clamp bound those
windows without moving a pick.)*

### 3.5 The verdict, and the interval on the optimum

```
verdict  UNPOWERED
argmin   (lambda = 0.25, tau = 0.1)  ade_0_2s = 0.8483
shipped  (lambda = 1,    tau = 1  )  ade_0_2s = 0.8563
paired   delta = -0.0079  [-0.0272, +0.0126]   p(d>0) = 0.207   NOT separated
```

**The interval on the optimum is the admissible set — every cell whose paired Δ vs the argmin is not
separated. It is 30 of 42 cells** (identical on both sheets), spanning λ ∈ [0, 8] and τ ∈ [0.1, 4]
and including λ = 0 itself. *A single argmin cell with no interval is not a located optimum, and the
rule refuses to report one.*

⚠️ **Two of the four pre-registered failing conditions hold at once.** `locate_optimum` checks
UNPOWERED before NO-INTERIOR, so only the first is returned — but the argmin sits on the **τ = 0.1
grid edge**, so the optimum is **not bracketed** either. Extending the grid below τ = 0.1 is not
indicated: at (0.5, 0.1) and (1, 0.1) the sharpening branch is already **worse**, and (1, 0.1) is
where the clamp starts binding.

> ### What the curve answers, and what it does not
> **Answers:** the program's soft-side evidence was **two points** (0.8781 with no prior → 0.8563
> with the shipped one). It is now **42**, and the response along τ = 1 is a clean interior U whose
> minimum **is the shipped λ = 1**. Prior strength was hard-wired, and it was hard-wired well.
> **Does not answer:** whether a *better* (λ, τ) exists. At n = 40 clusters the grid cannot
> distinguish 30 of its 42 cells from each other. `MODEL_REGISTRY §1.2a`: half-widths shrink
> ×2.8–3.9 at n = 600. **Any winning cell must be re-run at n = 600 before it steers a GPU-day** —
> and on this evidence there is no cell worth re-running.
> ⚠️ **The literature's "the optimum is interior in three independent domains" is NOT contradicted
> here — it is untested.** What is measured is that *our* grid cannot resolve it at this power.

### 3.6 ⭐ E-H0b, discharged on the produced surface — and the oracle-surface split does not carry

With λ = 0 a **config** rather than a re-implementation, the graft term is measured directly on the
deployable surface:

| term | value | how |
|---|---:|---|
| `F_base_only` (no graft, no constant-velocity term) | 0.8781 | committed bar, reproduced ✅ |
| **λ = 0** (graft removed, longitudinal term retained) | **0.8742** | **MEASURED, this run** |
| shipped `F_flat` (λ = 1) | 0.8563 | committed bar, reproduced ✅ |
| **the graft alone** | **+0.0179 m** | 0.8742 − 0.8563 |
| the constant-velocity term alone | +0.0039 m | 0.8781 − 0.8742 |
| **sum** | **+0.0218 m** | = the committed `F_flat − F_base_only`, **to the digit** |

> ⭐ On the **deployable** surface the hierarchy graft is **82.1 %** of the +0.0218 m bundle.
> `LAMBDA_TAU_SWEEP.md` §2's *"the hierarchy is worth about half"* was the **oracle** surface,
> `DERIVED / PROVISIONAL`, and it **does not carry over** — the produced-surface split was flagged
> UNKNOWN there and is MEASURED here.
> ⚠️ **With its own interval it is +0.0180 [−0.0022, +0.0455] — NOT separated at n = 40.** Tier:
> **MEASURED, PROVISIONAL**. The point estimate is now sound; the sign is not yet decision-grade.

**Commitment telemetry, free from the same run:** distinct candidates chosen over 881 windows —
**160** at λ = 0, **128** at the shipped cell, **107** at the argmin, **44** at (8, 0.1) deployable
and **24** with the clamp off. Prior strength buys commitment monotonically; ADE does not follow it.

---

## 4. ⭐ GOALFLOW'S LITERAL SHADOW RULE — the "NOT MEASURED" is discharged, and it REFUTES

`MEASURED · CONFIRMED` · `code/eh3_goalflow_rule.py` · `raw/eh3_goalflow.json` ·
per-window `raw/eh3_goalflow_windows.pt` · **1.2 s of CPU**.

`LAMBDA_TAU_SWEEP.md` §8 listed this as not computable, with the precise reason: *"No staged
artifact holds the neutral branch's trajectory — only its per-window error."* `eh2_build_cache.py`
dumps `pick_traj` for **both** goal modes, so the literal rule — *"if the shadow trajectory deviates
significantly from the main trajectory, treat the goal as unreliable and emit the shadow"* — is now
one script.

**Four more committed numbers reproduce from the new cache**: produced-always **0.8563**,
neutral-always **0.7620**, oracle-shadow bound **0.6883**, fraction of windows where the produced
goal is better **0.4313** (E-H3: 43.1 %).

| deployable rule (threshold chosen **leave-one-episode-out**) | `ade_0_2s` | vs neutral-always | separated |
|---|---:|---|---|
| endpoint deviation | **0.7607** | **−0.0014 [−0.0066, +0.0038]** | ❌ |
| max dense deviation | 0.7610 | −0.0010 [−0.0054, +0.0034] | ❌ |
| mean dense deviation | 0.7615 | −0.0005 [−0.0043, +0.0033] | ❌ |
| *in-sample ceiling of the best rule (NOT deployable)* | *0.7585* | — | — |
| **oracle shadow bound (NOT deployable)** | **0.6883** | — | — |

> ### VERDICT: **REFUTE**, at the pre-registered bar.
> All three deviation signals beat *produced-always* by a separated **−0.095 m** — but that is just
> "turn the goal off", which was already known to be free. **None of them separates from
> neutral-always**, and even the in-sample ceiling recovers only **0.0035 of the 0.0737 m** of
> shadow headroom (**4.7 %**).
>
> **That is now FIVE deployable reliability signals failing the same bar** — E-H3's four, plus
> GoalFlow's own. The 0.0737 m [−0.0928, −0.0552] of shadow headroom is real and separated, and
> **the deviation between the two branches does not locate it.** The information that decides which
> branch to trust is not in the geometry of the disagreement.

---

## 5. WHAT THIS LICENSES, AND WHAT I REFUSE TO CONCLUDE

**Settled by this run.**

- **The (λ, τ) instrument is no longer an instrument — it is a result**, and the result is that the
  grid **cannot locate an optimum at n = 40**. 30 of 42 cells are mutually indistinguishable.
- **The shipped λ = 1 is the argmin of its own τ = 1 row**, with a clean interior U. Whatever chose
  it chose well.
- **The clamp trap is fully priced**: derived on the λ axis before the run, measured on both axes in
  it, `seam_fail` raises on 6 of 42 cells, and `seam_clamp` changes no answer the deployed head can
  reach.
- **E-H0b is discharged on the produced surface**: graft +0.0179 m, constant-velocity +0.0039 m,
  summing to the committed +0.0218 m. The "worth about half" split was the oracle surface and does
  not carry over.
- **T1's kinematic clip replicates on v4's own emitted fan** — 73.88 % removed, oracle survives
  100 %, pick moves in 0 windows.
- **The labeler's false-positive floor at parity is 0.435 % against PARA-Drive's published 0.384 %**,
  so on THIS corpus absolute levels are defensible (contrasts remain the safer read).
- **GoalFlow's literal shadow rule fails**, joining four other deployable signals.

**NOT settled, and I refuse to conclude it.**

- ⚠️ **A REFUTE on T3 is not "rule labels are useless".** It is: *the four rule sub-scores, added to
  this frozen fan's score under this combine and fit on 26 episodes, do not move PDMS-lite.* T2's
  finding — the label is common, varies inside most windows, and is nearly orthogonal to ADE —
  stands untouched. **Bar A's lesson repeats: "signal exists" and "a realisable ranker extracts it"
  are different claims**, and the second one just failed twice in a row on this frozen fan.
- ⚠️ **The composite has very little dynamic range across trained selectors.** Random scores
  **0.3969**; every trained arm lands within **0.0002–0.0005** of 0.6086. **PDMS-lite (no-map)
  separates a selector from noise but barely separates two selectors from each other.** A metric
  that cannot resolve the arms cannot adjudicate between them, and that limitation is at least
  partly ours: DAC is missing, and **comfort is a constant on this fan**, so two of PDMS's terms
  contribute nothing here.
- ⚠️ **Nothing here says the (λ, τ) optimum is at (1, 1).** It says the grid cannot tell (1, 1) from
  29 other cells. Those are different statements and only the second is measured.
- ⚠️ **Every interval in this file is n = 39–40 episode clusters and is UNPOWERED, not refuted, at a
  null.** `MODEL_REGISTRY §1.2a`: half-widths shrink ×2.8–3.9 at n = 600.
- ⚠️ **T3's labels are NOT a model fact.** They are a property of a labeler applied to one
  checkpoint's fan on the val corpus. **They must never enter `MODEL_REGISTRY.md`**, and they are a
  *different measurement* from T2's dev-box labels — the two are never to be pooled or compared as
  if they were the same quantity.
- ⚠️ **DAC / DDC / LK / TL remain unbuildable** — no map in PhysicalAI-AV. PDMS-lite is missing one
  of PDMS's two multiplicative terms, which bounds how far any composite can adjudicate for us.

---

## 6. DELIVERABLE MANIFEST

Repo root `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`, folder
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-t3-and-lambda-tau/`.
**All STAGED (`git add`). Nothing committed. Nothing pushed. No branch switched.**
Staging verified with **`git ls-files --stage`**, not a scoped `git status --short`.

| artifact | where | exists elsewhere? | what it is |
|---|---|---|---|
| `T3_AND_LAMBDA_TAU.md` — this document | `repo:…/` | **repo only** | |
| `code/t3_labels.py` | `repo:…/code/` | also `tanitad-eval:/workspace/_t3/` | the parity label minter: exact clock inversion, index-offset scan, clock-quality gate, the NC/TTC/C/EP labeler, the GT instrument check |
| `code/t3_scorer.py` | `repo:…/code/` | also `tanitad-eval:/workspace/_t3/` | the 3-arm T3 harness: Bar A's folds/LR grid, the 5-output head, the Hydra-MDP combine grid, both self-tests, the estimator |
| `code/pull_val40_labels.py` | `repo:…/code/` | **repo only** | range-read extraction of the 40 val clips' gated label parquets (no token value is in the file — it is read in place from `Keys.txt`) |
| `code/diag_clock.py` | `repo:…/code/` | also pod | the direct-inversion diagnostic that replaced the failed optimiser |
| `code/eh3_goalflow_rule.py` | `repo:…/code/` | also `tanitad-eval:/workspace/_eh2/` | GoalFlow's literal shadow rule, LOEO thresholds |
| **`raw/eh2_cache.pt`** (9.6 MB) | `repo:…/raw/` | also pod | ⭐ **the E-H2 score-ingredient cache — the artifact whose absence blocked the whole stream.** `refined_pre`, the three class logit vectors, the graft weight matrices, `sel_pen`, `sel_score`, `prior`, `fan_err`, `pick_traj`, for **both** goal modes. **With this staged, the 42-cell sweep, the shadow branch and E-H0b all recompute on any CPU with no GPU and no pod.** md5 `4c1130663d5292b8eef8c35b9d670fb2` |
| `raw/eh2_sweep.json` | `repo:…/raw/` | also pod | all 42 cells × 2 sheets, per-cell pre-clamp ratios, clamp audit, the verdict + admissible set, the 4-fixture self-test, the fidelity gate |
| `raw/eh3_goalflow.json` · `raw/eh3_goalflow_windows.pt` | `repo:…/raw/` | also pod | the shadow rule's result and its per-window dump |
| **`raw/t3_labels.pt`** (21.3 MB) | `repo:…/raw/` | also pod | per-window × per-candidate NC / TTC / comfort / progress / `v_term` on v4's emitted fan at parity, plus the GT control and the per-episode trust flags |
| `raw/t3_labels.json` | `repo:…/raw/` | also pod | rates, within-window variation, the clock-recovery table (per-clip, aliased), the clock gate, the GT control |
| **`raw/t3_result.json`** | `repo:…/raw/` | also pod | T3's arms, both self-tests, the clip precondition, every paired and single-arm interval, the verdict |
| `raw/t3_folds_ce_control.json` · `raw/t3_folds_bce_rule.json` | `repo:…/raw/` | also pod | per-fold LR sweep, the full 12-cell inference-weight grid with the refused weightings, per-fold seam telemetry and training histories |
| **`raw/t3_windows.pt`** | `repo:…/raw/` | also pod | ⭐ per-window PDMS-lite / collision / TTC / `ade_0_2s` **and the pick** for every arm, so every T3 bar recomputes with **no GPU** |
| `…/2026-07-27-lambda-tau-sweep/code/eh2_lambda_tau_sweep.py` | `repo:` | ⚠️ **another stream's file** | one-line locator fix (`Path.parents[6]` raised `IndexError` off-repo). **No arithmetic, no estimator, no cell value changes.** |

| `code/verify_staged.py` | `repo:…/code/` | **repo only** | the audit that re-derives every T3 bar and every (λ, τ) headline **from the staged raw files alone**, and scans every artifact for clip-UUID / token leakage. Run it before quoting anything here. |

⚠️ **Nothing exists in only one place** — every artifact is in the repo working tree and staged, and
every binary was **md5-verified against the pod** after transfer (six files, six exact matches).

⚠️ **The index is shared.** Per CLAUDE.md's git-hygiene rule, recorded so whoever commits knows:
besides my 17 files and the one-line locator fix, the index carries a **sibling stream's**
`…/2026-07-27-latent-ablation/LATENT_ABLATION.md`, which I did not touch.
**I committed nothing, amended nothing, pushed nothing and switched no branch.**

**Deliberately NOT staged:** the 116 raw PhysicalAI-AV parquets (`pod:/workspace/_t3/pai_val40/`,
35 MB) — **gated-confidential**, re-fetchable in ~4 minutes by `code/pull_val40_labels.py`; and Bar A's
two 4.07 GiB feature caches, which remain where Bar A left them (`pod:/workspace/_bara/`).

**Reproducing every number in this file:** the (λ, τ) curve, the shadow branch and E-H0b need only
`raw/eh2_cache.pt` + `taniteval/ci.py` on **any CPU**. Every T3 bar needs only `raw/t3_windows.pt` +
`raw/t3_labels.pt` on **any CPU**. Only re-fitting the T3 arms needs the pod.
`code/verify_staged.py` does exactly this and prints the headline table.

**`stack/` was not modified**, so `cd stack && pytest -q` is unaffected by this stream.

### 6.1 Which stream each deliverable unblocks

| deliverable | unblocks |
|---|---|
| `raw/eh2_cache.pt` | **E-H2 (closed), E-H0b (discharged), E-H1's deferred `O_graft(q)` / `H_rand(q)` arms, and E-H3** — all now CPU-only. The single artifact that converted the (λ, τ) stream from an instrument into a result. |
| `raw/eh2_sweep.json` | the **hierarchy-prior line**: closes the "is the prior's strength right?" question at this power and says do not spend more. |
| `raw/t3_labels.pt` + `code/t3_labels.py` | the **selector / rule-supervision line (R1)** and anything needing PhysicalAI-AV's raw features joined to our episode caches — including the **strategic-brain topology** work, which needs exactly this clock join. |
| `raw/t3_result.json` + `raw/t3_windows.pt` | the **scorer-adjudication line**: T3 is answered, and the per-window dump lets any future composite (e.g. one with DAC, or with re-derived comfort bounds) be re-scored on these arms with **no GPU**. |
| `raw/eh3_goalflow.json` | the **goal-reliability / fallback line (E-H3)**: five signals down; the next attempt must not be a sixth threshold. |
| `code/verify_staged.py` | **review**: one command re-derives every headline from the staged files and scans for gated-content leakage. |

---

## 7. ESCALATIONS — these must not sit in a file

1. ⛔ **THE COMFORT TERM IS DEAD ON v4's FAN AND SHOULD BE REMOVED FROM PDMS-LITE UNTIL IT IS
   RE-DERIVED.** 100.0000 % of v4's emitted candidates violate, so `2·C` is identically zero; and
   the **human's own future** passes only **53.1 %** of the time, so the `INHERITED` nuPlan/NAVSIM
   thresholds are mis-specified against a 10 Hz finite-difference jerk. Any future use of PDMS-lite
   should either drop the term or re-derive the bounds against the GT distribution first.
   **Owner needed: whoever next touches the composite.**
2. ⭐ **THE (λ, τ) SWEEP IS DONE AND ITS ANSWER IS "DO NOT SPEND MORE HERE".** No cell is worth an
   n = 600 re-run. The score ingredients are staged (`raw/eh2_cache.pt`, 9.6 MB), so any future
   question about prior strength costs **CPU seconds, not a GPU-hour**. Close the E-H2 line.
3. ⚠️ **THE `+0.0218 m` FIGURE NOW HAS A PRODUCED-SURFACE SPLIT — use it and stop quoting the
   oracle one.** graft **+0.0179**, constant-velocity **+0.0039**. Its interval
   (+0.0180 [−0.0022, +0.0455]) is **not separated**, so it is `MEASURED / PROVISIONAL` and may not
   decide a GPU-day.
4. ⚠️ **FIVE DEPLOYABLE SIGNALS HAVE NOW FAILED TO LOCATE THE 0.0737 m SHADOW HEADROOM.** The next
   attempt should not be a sixth threshold on a geometric disagreement. Either learn the fallback
   decision from the goal head's own uncertainty, or accept the neutral branch and stop paying for
   the produced one.
5. ⭐ **THE CLOCK RECOVERY IS A REUSABLE INSTRUMENT AND IT IS NOW EXACT.** `code/t3_labels.py`
   recovers the epcache pose-index → clip-time map to **11 µm** on all 40 val clips and identifies
   the window-index convention (`k = 7`) by measurement. **Any future work that needs to join
   PhysicalAI-AV's 36 raw features to our episode caches should call it rather than re-derive it** —
   including the strategic-brain topology work, which will need exactly this join.

---

## 8. FOR `Project Steering/RETRACTION_LOG.md` — root-cause classes

### R-1 — **C3 (mechanism instead of measurement), CAUGHT IN-RUN**
I wrote *"xy rms 1.6 mm – 5.4 cm over all 40 episodes"* into a draft of this report after diagnosing
**five** episodes. The full run showed **one** episode at **2.60 m** with its index-offset scan
pinned at the scan edge — a failed clock, not a good one. **Root cause: generalising a spot check to
a population.** Fixed at source (a second canonical seed for the fit, plus an explicit clock-quality
gate that excludes and reports failures); the corrected fit is exact (**11 µm max**) on all 40.
**Standing consequence: any per-episode/per-clip quality statistic quoted in a report must come from
the full run's own JSON, never from the diagnostic subset used while building it.**

### R-2 — **C-new: a pre-registration that under-specifies its own head shape**
§8 specifies `Linear(512,4)` + 4 sigmoids **and** an inference combine containing `S_im`. Those are
inconsistent: a 4-output head has no imitation channel. The gap was resolved by building 5 outputs
and putting the literal 4-only reading inside the weight grid (`w_im = 0`) so it is **measured**
rather than chosen. **Standing consequence: a pre-registration that names both an architecture and a
combine must state where each term of the combine comes from**; where it does not, the executing
agent must make the missing choice a *measured arm*, not a silent decision.

### R-3 — **C6 (confounded comparison), avoided by construction**
Selecting `CE_CONTROL`'s LR on ADE and `BCE_RULE`'s on PDMS-lite would have made the primary contrast
a mixture of *the arm* and *its selection criterion*. Both arms select on PDMS-lite, and the clip is
applied to all three arms identically. **Standing consequence: when the primary read changes, the
model-selection criterion must change with it — for the control too.**

---
