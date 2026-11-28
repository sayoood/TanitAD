# E-GOAL-2 — PRE-REGISTRATION

**Written BEFORE any number in `EGOAL_2.md` was computed.** Nothing here is edited after the first
placement is run; corrections are appended in §9 with a timestamp and nothing is deleted.

⚠️ **Date note.** The deliverable folder is named `2026-07-28-…` as briefed; wall-clock at
authoring is **2026-07-27** (the program's narrative clock runs ahead — known loop artefact).
Every artifact timestamp below is real wall-clock.

---

## 0. The question, in one line

**Does E-GOAL-1's `+23.6 %` recovery of the fan's headroom from ego-kinematics survive a
15× increase in placement episodes (40 → 600), on BOTH resamplers?**

E-GOAL-1 (`…/incoming/2026-07-27-egoal-1-lead-vehicle/EGOAL_1.md §0.3`) is **PROVISIONAL** for one
stated reason: the `iid` resampler separates with an upper bound of **−0.0008** (a hair from zero)
and the conservative `by_speed` resampler gives the same sign and size (**+20.3 % / +22.9 %**) but
does **not** separate. `MODEL_REGISTRY.md §1.2a` measures CI half-widths shrinking **×2.8–3.9
(mean 3.4)** going 40 → 600 episodes. **This is a power upgrade, not a new experiment.**

## 1. ⛔ WHAT MAY CHANGE AND WHAT MAY NOT

| component | status |
|---|---|
| the along-track residual **pools** (`E1_ego`, `E1_L`, `CV`, `E0_v0`, `P_ORACLE`, …) | ⛔ **FROZEN** — read verbatim from `…/2026-07-27-egoal-1-lead-vehicle/raw/eg_oof_pred_gbm.npz`. Not re-fit. Not re-tuned. |
| the injection construction (`goal = [g_true_along + resampled_residual, head_cross]`) | ⛔ **FROZEN** — `eg_place.py::realise` / `goal_reference` / `pick_nearest_to` are **imported**, not re-implemented |
| both resamplers (`iid`, `by_speed`), 10 shared speed deciles, `N_SEEDS = 16`, seeds `1000+s` | ⛔ **FROZEN** |
| estimator: paired episode-cluster bootstrap, `taniteval/ci.py`, **B = 2000**, unit = **episode** | ⛔ **FROZEN**. `overlapping_holdout_se` is **never called** |
| **`n` — the placement window/episode set** | ✅ **THE ONLY REGISTERED CHANGE**: 881 windows / 40 episodes → the 600-episode `physicalai-val-0c5f7dac3b11` build |
| the **cross-track background** `head_cross` | ⚠️ **must be re-derived at n = 600 and CANNOT be identical** — see §4. This is declared here, in advance, with its own fidelity bridge. |

**Any other change is an ADDITIONAL ARM, reported beside the registered one, never instead of it.**
Re-fitting the head to chase a bar is the forking-paths failure `GATE_PROTOCOL §0.3` forbids and is
not done.

## 2. Pre-registered decision rule

| verdict | condition |
|---|---|
| **CONFIRM** | at n = 600, `E1_ego`'s paired recovery vs as-trained is **separated-better** **AND** the conservative `by_speed` resampler **also separates** ⇒ tier → **CONFIRMED**; the speed-history feature enters the v5 selector |
| **PARTIAL** | `iid` separates, `by_speed` does not ⇒ report **both**, name which v5 should be sized against, and state plainly that the conservative estimate is the one to plan with |
| **REFUTE** | the effect does not survive n = 600 (not separated, or separated-worse, on the registered `iid` arm) ⇒ **the +23.6 % was an n = 40 artifact. Say so plainly. No re-scoping.** This retires the program's only positive selection lever. |

**Primary arm = `E1_ego|iid`** (ego kinematics, the registered lever). `E1_L` is reported beside it
but the lead block is already CONFIRMED-REFUTED by E-GOAL-1 and is not the question here.

### 2.1 ⚠️ What input would make each rule return a FAILING value — and the proof it can

| rule | the input that makes it fire | can it fire? |
|---|---|---|
| **CONFIRM** | an along-track error structure that genuinely helps the REF-C pick | **YES — proven by `P_ORACLE`** (future speed at t+2 s). It recovered **+59.8 %** separated at n = 40; at n = 600 it must remain separated-better. **If it does not, the instrument cannot return CONFIRM and the run is VOID, not negative.** |
| **REFUTE** | an error structure that carries no usable information | **YES — proven by two controls.** (a) `E0_v0` / `CV` / the **parent's own residual** were all separated-**WORSE** at n = 40 (−66.1 % / −57.7 % / −33.1 %); at n = 600 they must stay separated-worse. (b) ⭐ **`E1_noise_hist`** (§5) — the deliberately failing input. |
| **PARTIAL** | `iid` separates, `by_speed` does not | it is the arithmetic middle and needs no separate proof; both bounding controls above are what make it meaningful |

**Both controls are mandatory and are reported whatever the verdict.** A run in which either
misbehaves is reported as **VOID**, not as a result.

## 3. ⛔ S0 — THE LEAK / OVERLAP CHECK. Priority 1: a leaked result is worse than no result

A sibling found **4 of 36 val episodes would have leaked (11 %)** in an adjacent cache that carried
no usable `episode_id`, and caught it **only by fingerprinting episodes by a hash of their poses**.
The same check runs here, on the build the fan dump actually reads, **before any fan is produced**:

1. **sha256 over the raw `poses[T,4]` float32 bytes** of every episode in
   `pod2:/root/valdata/physicalai-val-0c5f7dac3b11` (600) and every episode in the parity train
   corpus `pod2:/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894` (2376).
   **`episode_id` is NOT a key — it collides** (the parity train holds 2376 episodes and only 2342
   unique ids).
2. **Report the overlap count even if it is zero.**
3. **Order-preserving-prefix check**: `published40[i] == val600[i]` for every `i ∈ [0,39]` by the
   same fingerprint ⇒ 40 → 600 **adds** episodes and **re-selects none**. If this fails, parity is
   broken and the run is **REFUSED**.
4. **Cross-copy identity**: `/root/valdata/…` (what the dump reads) vs
   `/workspace/data/physicalai_phase0/_epcache/…` (what the sibling audited) must be the same 600
   fingerprints. A prior audit of a *different copy* is not an audit of this one.

⛔ **Any non-zero train overlap ⇒ the affected episodes are DROPPED and the drop is reported**;
if the prefix property fails, the stream **REFUSES** rather than re-selecting.

## 4. ⚠️ THE ONE THING THAT CANNOT BE HELD IDENTICAL — declared in advance

E-GOAL-1's placement holds the **cross-track** coordinate at the parent's learned value
(`gi_head_preds.npz::best` = `H_ridge_all_raw`, cross-track MAE **0.400 m**). That head's feature
set is `F_lat` ⧺ `F_ego` ⧺ `F_ans` ⧺ `PCA32(F_conf_raw)`, and **`F_lat` / `F_conf_raw` come from
`eh2_cache.pt` — v4 latent-derived ingredients that exist ONLY on the canonical 881 windows.**
Reproducing them at 600 episodes requires a second model (v4) and a second GPU pass.

**Registered decision, made before any placement number was computed:**

- The n = 600 cross-track background is a **ridge on the fan-only blocks `F_ego` ⧺ `F_ans`**, with
  the **identical** ridge machinery (`ALPHAS = logspace(-3,4,22)`, inner `GroupKFold` over
  train-fold episodes, 5 episode-disjoint outer folds, `StandardScaler`, target `raw`). Only the
  two eh2 blocks are dropped, because they do not exist at n = 600.
- ⭐ **FIDELITY BRIDGE `F-B2`, and it can fail:** the *same reduced* cross-track is substituted into
  the **n = 40** placement. If the published recoveries (**+23.6 % / +25.9 % / −33.1 % / −66.1 %**)
  do not reproduce **within ±3 recovery points and without any sign or separation flip**, the
  substitution is declared **UNSAFE**, the n = 600 numbers are reported as **UNVERIFIED**, and the
  correct fix (build `eh2` at 600) is escalated rather than papered over.
- The cross-track is **identical across every arm** in a given run, so it is a shared background,
  not a treatment. It never enters a paired contrast asymmetrically.

## 5. Arms — the registered set, plus two declared additions

| arm | source of the along-track residual pool | role |
|---|---|---|
| **`E1_ego`** | frozen pool, E-GOAL-1 | ⭐ **THE REGISTERED PRIMARY** |
| `E1_L` | frozen pool | reported beside it |
| `E1_L_X`, `E1_L_X_D` | frozen pools | reported |
| `E0_v0`, `CV` | frozen pools | **negative bounds** — must stay separated-WORSE |
| `P_ORACLE` | frozen pool | **positive control** — must stay separated-BETTER |
| `PARENT_RESAMP` | the parent head's own along residual, identical resampler | **decorrelation control** — must stay separated-WORSE (−33.1 % at n = 40) |
| ⭐ **`E1_noise_hist`** *(ADDITION, declared here)* | E-GOAL-1's `E1_ego` feature set with `dv_0p5`, `dv_1p0`, `v_lag_0p5`, `v_lag_1p0` **replaced by Gaussian noise matched in mean and SD**, re-fit with the **identical** GBM hyper-parameters on the **identical** 612-clip windows and folds | ⛔ **THE DELIBERATELY FAILING INPUT.** A pure-noise "history" must recover **nothing beyond the no-history ablation** (R3: 1.0733 m). If noise-history recovers like real history, the +23.6 % is a fitting artifact and the whole stream is VOID. |
| `E1_nohist` *(ADDITION)* | E-GOAL-1's R3 ablation, re-fit identically | the honest reference the noise arm must land on |

## 6. Axes — and the guard that cannot fail at small n

⚠️ E-GOAL-1's positive control `P_ORACLE_STRONG` **passed decisively on MAE (−61.2 %) and failed to
separate on the along-track RMS axis** at n = 612 clips ⇒ *no non-separation on that axis is
evidence of absence* (class `SEPARATION-CLAIMED-ON-AN-UNPOWERED-AXIS`).

**Stated in advance so it cannot be spun afterwards:** the n that this stream raises is the
**placement episode count** (40 → 600). The head-fit RMS/MAE axis is a property of the **dev-box
612-clip fit, which is FROZEN and therefore NOT re-powered by anything done here.** Both are
reported and the distinction is stated in the headline:

- **The verdict axis** is the realised `ade_0_2s` recovery — a **mean-over-waypoints of L2**, i.e.
  an **MAE-family** statistic. It is what E-GOAL-1's §0.3 decides on and it is the primary here.
- **Beside it**, an **RMS-family** companion (root-mean-square of the per-window realised error) is
  reported for every arm, so the reader sees whether the tail-dominated axis behaves differently at
  n = 600.
- The frozen head-fit RMS/MAE numbers are quoted as `INHERITED` from E-GOAL-1 and **explicitly
  marked NOT re-powered**.

## 7. Estimator, fixed in advance

Paired **episode-cluster bootstrap**, `taniteval/taniteval/ci.py`, **B = 2000**, resampling unit =
the **val episode**. Separation predicate: the CI excludes 0. `overlapping_holdout_se` is **never
called** — it biases the point estimate as well as the interval (CLAUDE.md, measured over 27 arms).
Not separated ⇒ **UNPOWERED NOT REFUTED**, quoted with the n at which it would separate.

Every interval reports **n windows AND n episode clusters** (`MODEL_REGISTRY §1.2a` rule 2).

## 8. Corpus, parity, hosts

- ⛔ **Parity is sacred.** `physicalai-train-e438721ae894` (2376 eps, skip-hash `f09e44db`) is
  **read only to fingerprint it for the leak check** and is never re-selected. `_epcache` is never
  written. Anything that would re-select episodes is **REFUSED**.
- **Host: `tanitad-pod2`** — it is the only host holding the 600-episode `physicalai-val-0c5f7dac3b11`
  build *and* the parity train corpus. ⛔ **pod1 is TRAINING and is not touched.** Verified idle
  before loading: GPU 0 MiB / 46068 MiB, no trainer or build process. The REF-C-XL weights are
  pulled from `Sayood/tanitad-refc-xl` (HF, ~246 MB/s from a pod) rather than relayed through the
  ~1 MB/s dev box.
- ⚠️ **The dev box's own episode cache is keyed `14231cd29c74`, NOT parity `e438721ae894`** — no
  parity-dependent step runs there. The dev box is used only for the frozen residual pools and the
  CPU placement arithmetic.

⭐ **FIDELITY GATE `F-A`, and it can fail:** the 600-episode fan dump's **first 40 episodes must
reproduce the committed 881-window `fan_refc-xl-30k.pt` element-for-element** (same `sel`, same
`fan`, same `gt`, same `v0`). The published 40 are `val600[0:40]` in order, the decode is
deterministic, and the model call is the same one. **If the prefix does not reproduce, the 600-ep
dump is wrong and nothing downstream may be quoted.** *(Three streams in two days found bugs in
their own scoring code producing stable, plausible, wrong numbers. This gate exists because of
them.)*

⭐ **FIDELITY GATE `F-B`:** before any n = 600 number is quoted, the **n = 40** placement is re-run
with this stream's code and must reproduce `EGOAL_1.md §0.3` to **four decimals** (+23.6 %, +25.9 %,
−33.1 %, −66.1 %, −57.7 %, +59.8 %). Any mismatch aborts.

## 9. ⚠️ AMENDMENTS — appended after the first run, timestamped, nothing deleted

*Every clause in §§0–8 stands as written. These are additions, and the reader can see exactly what
changed and when.*

**A1 (2026-07-27, after the n = 40 bridge).** ⛔ **The §4 bridge control `F-B2` FAILED, as it was
written to be able to.** The registered `reduced` cross-track substitute deviates **+5.6 recovery
points** on `E1_ego|iid` (+29.2 % vs the published +23.6 %) and **flips `E1_ego|by_speed` from
not-separated to separated at n = 40, before `n` changes at all.** Per §4 it is therefore declared
**UNSAFE as a drop-in** and is **not** used to carry E-GOAL-1's numbers to n = 600. It is reported
in `EGOAL_2.md §4.2` rather than quietly used.

**A2 (2026-07-27, after A1).** Two further cross-track backgrounds were added **as declared
additional arms, not as replacements**, because A1 showed the background is load-bearing and one
substitute is not enough to characterise it:
- **`sel`** — the selector's own 2 s endpoint. **Zero fitting, zero folds, zero hyper-parameters**,
  therefore *provably* identical in construction at n = 40 and n = 600.
- ⭐ **`parent_resampled`** — the parent head's **own 881 cross-track residuals**, resampled i.i.d.
  onto the true cross-track by the construction's existing machinery, **drawn once per seed from an
  independent rng stream and shared identically across every arm**. This carries E-GOAL-1's
  cross-track *error structure* (MAE 0.400 m) to any n.

**Registered before the n = 600 fan finished decoding:** ⭐ **`parent_resampled` is the PRIMARY
carrier** for the n = 600 verdict. Reason, stated in advance: it is (a) the only background that
reproduces E-GOAL-1's cross-track error magnitude, (b) the **conservative** one of the three, and
(c) **not separated at n = 40** — so n = 600 genuinely decides it instead of confirming something
a background choice already decided. `sel` and `reduced` are **secondary** and test stability.

**A3 (2026-07-27).** ⚠️ **The comparison is re-expressed at the level at which it is actually
valid.** Because no n = 600 background reproduces the parent's head, an n = 600 number **may not be
compared to E-GOAL-1's +23.6 % / +20.3 %.** The pre-registered "only `n` changes" contrast is
therefore read **within construction**: n = 40 vs n = 600 **under the identical background**. Both
cells exist for all three n-portable backgrounds. This is a **stricter** reading of §1, not a weaker
one — it removes a confound §1 did not anticipate.

**A4 (2026-07-27, at S1).** `taniteval.refc_rerank.dump` hard-codes its val path and output. Rather
than edit a committed module, `code/e2_dump600.py` re-points two module constants and calls
`refc_rerank.dump()` **unmodified**, so the decode is the same one that produced the committed
40-episode fan. Its md5 on pod2 was verified identical to the repo copy first.
