# E-GOAL-2 — does the ego-kinematics goal lever survive n = 600?

**Stream:** `2026-07-28-egoal-2-power` · **Hosts:** dev box (CPU) + `tanitad-pod2` (A40, verified idle)
**Wall-clock date:** 2026-07-27 *(folder named `2026-07-28-…` as briefed; the program's narrative
clock runs ahead of wall-clock — flagged, not silently absorbed).*
⛔ **pod1 was never contacted** (it is training). pod3 was probed read-only once; the eval pod was
used read-only once, to fingerprint the published 40-episode val build.
**Pre-registration:** `PRE_REGISTRATION.md`, written **before any number below was computed**;
amendments appended in its §9, nothing deleted.
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **the val episode**. `overlapping_holdout_se` is **never called**.
**Total compute:** 19 min A40 + 6 min pod CPU + ~25 min dev-box CPU.

---

## 0. HEADLINE

> ### ⭐⭐ **CONFIRM. At n = 600 the ego-kinematics goal head recovers `+25.4 %` of the fan's headroom on the CONSERVATIVE carrier — `−0.0784 [−0.0960, −0.0606]` — and `by_speed`, the resampler that would not separate at n = 40, now separates at `+26.2 % [−0.0987, −0.0631]`. Both resamplers, all three cross-track backgrounds, every control behaving. The pre-registered CONFIRM condition is met and the tier goes to CONFIRMED.**

> ### ⛔ **AND THE PREDICATE THAT SAID SO IS NO LONGER DISCRIMINATING AT n = 600 — a deliberately information-free arm separates too (+9.1 %). The claim survives only because a DIRECT contrast was measured: history vs the same columns filled with noise is `−0.0504 [−0.0519, −0.0490]`, while dropped-history vs fake-history is `−0.0001 [−0.0006, +0.0004]`, a clean null. ⭐ 64 % of the recovery is speed history.**

> ### ⚠️ **A THIRD FINDING, AND IT RE-SCOPES E-GOAL-1's NUMBER: at FIXED n and FIXED along-track error, the recovery spans `+13.3 % … +29.2 %` purely on which learned cross-track sits in the background — and `E1_ego`'s separation flips inside that range. E-GOAL-1's `+23.6 %` is a mid-range value conditional on an unreported background, not a property of the lever alone.**

**⇒ The speed-history feature ENTERS the v5 selector.** Plainly stated, with its limit in §7.

---

## 1. ⛔ S0 — THE LEAK / OVERLAP CHECK. Priority 1, and it is CLEAN

`raw/e2_leak.json` (pod2) · `raw/e2_pub40_fingerprint.json` (eval pod) · `code/e2_leak.py`

**Method: sha256 over the raw `poses[T,4]` float32 bytes**, `torch.load(mmap=True)`. `episode_id` is
**not** a usable key — the parity train corpus holds **2376 episodes and only 2342 unique
`episode_id`s** (34 collisions), so an id check over-reports; filenames collide completely across
caches, so a name check under-reports.

| check | required | measured |
|---|---|---|
| **A — val600 × parity train `physicalai-train-e438721ae894` (2376 eps)** | overlap = 0 | ⭐ **`overlap_n` = 0 / 600 · 0.0000 %** ✅ |
| val600 internal collisions | 0 | **600 unique sha256 of 600** ✅ |
| parity-train internal collisions | *(reported)* | **2376 unique sha256 of 2376** |
| **the check the fingerprint replaces** — filename overlap | — | ⚠️ **600 / 600 = 100 %.** A name-based check would have called this a total leak. **The fingerprint is doing the work, not decorating it.** |
| **B — order-preserving prefix**, `published40[i] == val600[i]` ∀ i ∈ [0,39] | must hold or REFUSE | ✅ **40/40 sha-identical, positionally, ACROSS TWO HOSTS** (eval pod's own 40 vs pod2's 600) |
| **C — copy identity**: the build the dump *reads* (`pod2:/root/valdata/…`) vs the `_epcache` copy a sibling audited | must be the same 600 | ✅ **SAME BUILD** — 600/600 shared, positionally identical |

> ⭐ **0 episodes leak. Reported because it was run, not because it came back clean.**
> ⭐ **Check C is the gap that was actually open.** The sibling's 2026-07-26 audit
> (`…/2026-07-26-pod2-eval-host/artifacts/prefix_disjointness_result.json`) fingerprinted the
> **`_epcache`** copy. The fan dump reads **`/root/valdata/…`** — *a different path on the same pod*.
> **An audit of a different copy is not an audit of this one.** *(Class `ABSENCE-FOUND-AT-ONE-LOCATION`.)*
> ⛔ **Parity holds:** 40 → 600 **adds** 560 episodes and re-selects none. `_epcache` never written.

---

## 2. ⭐ S1 — THE FAN AT n = 600, and a fidelity gate that FAILED and then isolated its own cause

### 2.1 Host, and a checkpoint that was not where one `ls` said it was

⚠️ **`/root/models` on pod2 holds only `flagship-30k`** — on that evidence the REF-C-XL weights were
absent and a 3.02 GB cross-pod transfer was required. **A second probe found them:
`/workspace/models/refc-xl-30k/ckpt.pt`, md5 `966d4eff1ea5ddf86efba01b8344e198` — matching the
registry and the HF-push record byte-for-byte.** ⇒ **the transfer was unnecessary.**
*(Class `ABSENCE-FOUND-AT-ONE-LOCATION`, twice in one stream.)*

Pod state before loading: GPU **0 MiB / 46068 MiB**, no trainer or build process, disk checked with a
**real 500 MB `dd` write (502 MB/s)** — never `df`.

**Harness drift check — pod2 runs the COMMITTED decode:** `refc_rerank.py` `d59e82e8…`,
`data.py` `01c94d6d…`, `ci.py` `ef925f06…`, `loaders.py` `0cf85a29…` — **all four md5-identical to
the repo.** `code/e2_dump600.py` re-points two module constants and calls `refc_rerank.dump()`
**unmodified**; the repo module is not edited.

**Result:** `13,198 windows × 256 anchors / 600 episodes` in **1147 s (19 min)** of A40.
The window count matches `MODEL_REGISTRY §1.2a`'s 600-episode deployment (**13 198**) exactly.

### 2.2 ⛔ F-A failed as written — and the discriminating experiment says why

The registered gate was *"the 600-episode dump's first 881 rows reproduce the committed
`fan_refc-xl-30k.pt`."* **It did not:** `fan` max |Δ| **0.0185 m**, `logits` **0.106**, `sel`
differing on some windows. `gt`, `cv`, `v0`, `speed`, `head_deg`, `a_gt`, `v_target` were **exactly
0.0**, so the *window alignment* was provably right and the difference was in the decode itself.

**Two hypotheses, one 57-second experiment: re-dump 40 episodes on pod2 with the same code.**

| comparison | what it isolates | result |
|---|---|---|
| **pod2-600[0:881] vs pod2-40** | does **`n`** change the decode? | ⭐ **max \|Δ\| = 0.0 on EVERY field** — `fan`, `logits`, `sel`, `gt`, `cv`, `v0`, `speed`, `head_deg`, `a_gt`, `v_target`. **The 600-episode dump CONTAINS the 40-episode dump bit-for-bit.** |
| **committed-40 (eval pod, 2026-07-20) vs pod2-40 (today)** | host / code vintage | the **same** 0.0185 m / 0.106 / `sel` differences ⇒ **the divergence is entirely host-or-vintage and has nothing to do with `n`** |

⇒ **F-A passes in the form that matters**, and the failure was informative rather than fatal.
`raw/e2_fanmatch_pod2_40_vs_600.json`, `raw/e2_fanmatch_committed_vs_pod2_40.json`.

⚠️ **Consequence, adopted before any n = 600 number was quoted:** the n = 40 cells are **re-run on
the pod2 fan**, so the n = 40 → n = 600 contrast differs in **literally nothing but `n`, at bit
level**.

### 2.3 🟠 A reproducibility fact worth escalating on its own

**The committed `fan_refc-xl-30k.pt` is not bit-reproducible** from the current committed code, on a
current pod, with the same checkpoint md5 and the same module md5s. **Quantified rather than
alarmed about:**

| | value |
|---|---:|
| windows where the selected anchor differs | **1 of 881 (0.11 %)** |
| as-trained `ade_0_2s` | **0.4714 → 0.4712**, paired **−0.0002 [−0.0006, 0.0000] — NOT separated** |
| `oracle_in_fan` | 0.1640 → 0.1639 |
| fan mean \|Δ\| / p99 \|Δ\| | 0.000233 m / 0.0017 m |
| `nav_mode` provenance | committed `None` · today `follow_constant` |

⇒ **float non-determinism, not a code change.** ⭐ **This is also the first direct evidence FOR the
claim `refc_rerank.py` makes in a comment — that `follow_constant` == the historical `nav_cmd=None`**
— which until now was asserted, not measured. **The committed 0.4714 should be quoted as
`0.4714 (reproduces to 0.4712, Δ −0.0002, not separated, on a different host)`.**

---

## 3. ⭐ F-B — THE SCORING-CODE GATE. E-GOAL-1 reproduces to 0.010 recovery points

`raw/e2_place_n40_parent.json`

> *"Three streams in two days found bugs in their own scoring code producing stable, plausible,
> wrong numbers. Assume yours has one until a fidelity control says otherwise."*

Before any n = 600 number existed, this stream's engine was run on the **committed 881-window fan
with E-GOAL-1's own cross-track background** and compared to
`…/2026-07-27-egoal-1-lead-vehicle/raw/eg_place.json` — **the raw JSON, never the doc's rounded
table.**

| quantity | E-GOAL-1 | this stream |
|---|---:|---:|
| as-trained `ade_0_2s` / `R_goal2s` / `oracle_in_fan` / headroom | 0.4714 / 0.2009 / 0.1640 / 0.2705 | **identical to 4 dp** |
| **max \|Δ recovery\| over 16 arm×resampler cells + both control cells** | — | ⭐ **0.010 recovery points** |
| the family-matched requirement curve σ₀ / σ₅₀ | 1.1434 / 0.5907 m | ⭐ **1.1434 / 0.5906 m** |

`E1_ego|iid` returns **+23.6 %, −0.0638 [−0.1271, −0.0008]** — reproducing even the **−0.0008**
upper bound that made E-GOAL-1 PROVISIONAL. `realise` / `pick_nearest_to` / `goal_reference` are
**imported** from `eg_place.py`, not re-implemented.

---

## 4. ⛔⛔ THE REGISTERED BRIDGE FAILED — and the reason re-scopes E-GOAL-1's headline

### 4.1 What could not be carried to n = 600

E-GOAL-1 holds the **cross-track** coordinate at the parent's learned head `H_ridge_all_raw`, whose
features include `F_lat` and `PCA32(F_conf_raw)` — **v4-latent blocks from `eh2_cache.pt`, which
exists only on the canonical 881 windows.** `PRE_REGISTRATION §4` declared this in advance, named a
`reduced` ridge on the fan-only blocks as the substitute, and registered the condition that would
make it inadmissible: **±3 recovery points, no sign or separation flip.**

### 4.2 It blew the tolerance — in BOTH directions

Identical along-track pools, identical resamplers, identical seeds, **identical n = 40. Only the
cross-track background changes.**

| cross-track background | cross MAE | **`E1_ego\|iid`** | **`E1_ego\|by_speed`** | decorrelation control |
|---|---:|---|---|---|
| **`parent`** — E-GOAL-1's own (`H_ridge_all_raw`, eh2) | 0.400 m | **+23.6 %** ✅ sep [−0.1271, −0.0008] | **+20.3 %** ❌ not sep | −33.1 % ✅ worse |
| `reduced` — ridge on `F_ego` ⧺ `F_ans` *(the registered substitute)* | 0.303 m | **+29.2 %** ✅ sep | **+26.6 %** ✅ **sep — FLIP** | +13.8 % ❌ **not worse — FLIP** |
| `sel` — the selector's own endpoint, **zero fitting** | 0.307 m | **+28.7 %** ✅ sep | **+25.9 %** ✅ **sep — FLIP** | −40.4 % ✅ worse |
| ⭐ `parent_resampled` — the parent's own cross **residuals**, resampled | 0.400 m | **+13.3 %** ❌ **not sep — FLIP** | **+10.9 %** ❌ not sep | −41.6 % ✅ worse |

*(All four rows are on the **committed** 881-window fan, so the first row is a direct comparison to
E-GOAL-1's published numbers — which it reproduces exactly. §6 uses the host-matched pod2 fan.)*

> ⛔ **The registered substitute is inadmissible: +5.6 recovery points and a `by_speed` separation
> flip, at n = 40, before `n` changes at all. Using it at n = 600 would have MANUFACTURED a CONFIRM
> out of a background choice.** It is reported and discarded, not used. *(This is the forking-paths
> failure `GATE_PROTOCOL §0.3` forbids, caught by a control registered in advance to catch it.)*
>
> ⭐⭐ **The general fact is bigger than the bridge. At fixed n and fixed along-track error
> structure, the recovery spans +13.3 % → +29.2 % — a 15.9-point range — purely on which learned
> cross-track sits in the background, and `E1_ego|iid`'s separation flips inside that range.**
> ⇒ **E-GOAL-1's `+23.6 %` is a mid-range value CONDITIONAL ON A BACKGROUND IT DID NOT REPORT**, and
> its `by_speed` non-separation is **not a property of the along-track lever alone.**
> *(New retraction class, §8: `RECOVERY-CONDITIONAL-ON-AN-UNREPORTED-BACKGROUND`.)*

### 4.3 ⇒ How n = 600 must be read — decided before the 600 numbers existed

**No n = 600 number may be compared to E-GOAL-1's +23.6 % / +20.3 %**, because no n = 600 background
reproduces the one those numbers were measured against. The pre-registration asks for the contrast
**in which only `n` changes**, so the verdict is read **within construction**, against this stream's
own n = 40 row **under the identical background and on the identical host**. Three backgrounds are
n-portable; all three are run at both n.

⭐ **`parent_resampled` is the registered PRIMARY carrier** (`PRE_REGISTRATION §9 A2`): it carries
E-GOAL-1's **cross-track error structure** (the parent head's own 881 cross residuals, MAE 0.400 m,
resampled by the construction's own machinery, drawn once per seed and **shared identically across
every arm**), it is the **conservative** one, and **it is NOT separated at n = 40** — so **n = 600
genuinely decides it** rather than confirming something a background already decided.

---

## 5. ⭐ BOTH-DIRECTIONS VALIDATION — and the failing input really fails

`raw/e2_arms.json` · `raw/e2_extra_pools.npz` · `code/e2_arms.py`. E-GOAL-1's fitter
(`eg_fit.fit_predict`), folds, hyper-parameters and seed are **imported**, on the identical
99,935 windows / 612 clips.

| arm | along RMS | along MAE | required | |
|---|---:|---:|---|---|
| **`E1_ego_REPRO`** — E-GOAL-1's primary, **re-fit from the parquet** | **0.9305** | 0.4925 | must reproduce the frozen pool | ⭐ **max \|Δpred\| = 0.0 — BIT-IDENTICAL** ✅ |
| `E1_nohist` — history columns **dropped** (E-GOAL-1's R3) | **1.0733** | 0.5960 | must reproduce R3's 1.0733 | ✅ exact |
| ⛔ **`E1_noise_hist`** — the four history columns **replaced by Gaussian noise matched to each column's own mean and SD** | **1.0737** | 0.5953 | must recover **nothing** beyond `E1_nohist` | ⭐ **0.0004 m from the ablation, 0.1432 m from real history** ✅ |

> ⭐ **The noise arm holds MODEL CAPACITY FIXED and removes only information** — dropping columns
> changes both. **A pure-noise "history" buys 0.3 % of what real history buys.**

---

## 6. ⭐⭐ THE n = 600 PLACEMENT

`raw/e2_place_n600_*.json` · `raw/e2_place_n40p2_*.json` · `raw/e2_summary.json` · `code/e2_summary.py`

### 6.1 The deployment — and it is NOT uniformly "easier"

| | n = 40 (pod2 fan) | **n = 600** |
|---|---:|---:|
| windows / **episode clusters** | 881 / **40** | **13 198 / 600** |
| as-trained `ade_0_2s` (`a0`) | 0.4712 | ⚠️ **0.5015** |
| true-2 s-goal realised (`R_goal2s`) | 0.2009 | **0.1933** |
| `oracle_in_fan` | 0.1639 | 0.1547 |
| **headroom** (`a0 − R_goal2s`) | 0.2703 | **0.3082** |

> ⚠️ **`MODEL_REGISTRY §1.2a` records the 600 build as an EASIER corpus (CV floor 0.8377 → 0.6917,
> v1's flagship 0.4271 → 0.4108). REF-C-XL's selector goes the OTHER WAY: 0.4712 → 0.5015.**
> ⇒ **"the 600 deployment is easier" is arm-dependent, not a corpus property**, which sharpens
> registry rule 1 (*never mix deployments*) rather than contradicting it.

### 6.2 ⭐ THE PRIMARY RESULT — `parent_resampled`, the conservative carrier

Paired episode-cluster bootstrap vs as-trained, B = 2000, unit = episode. **Only `n` differs between
the two columns.**

| arm | mode | **n = 40** | **n = 600** | CI half-width shrink |
|---|---|---|---|---:|
| ⭐ **`E1_ego`** | `iid` | +13.3 % · −0.0358 [−0.1028, +0.0311] ❌ **not sep** | ⭐ **+25.4 % · −0.0784 [−0.0960, −0.0606] ✅ SEPARATED** | **×3.78** |
| ⭐ **`E1_ego`** | `by_speed` | +10.9 % · −0.0294 [−0.0956, +0.0384] ❌ **not sep** | ⭐ **+26.2 % · −0.0807 [−0.0987, −0.0631] ✅ SEPARATED** | **×3.76** |
| `E1_L` | `iid` / `by_speed` | +15.7 % / +12.5 % ❌ | **+27.5 % / +28.0 % ✅** | ×3.82 / ×3.71 |
| `E1_nohist` | `iid` / `by_speed` | −5.6 % / −8.8 % ❌ | **+9.1 % / +10.4 % ✅** | ×3.84 / ×3.66 |
| ⛔ `E1_noise_hist` | `iid` / `by_speed` | −5.6 % / −8.3 % ❌ | **+9.1 % / +10.4 % ✅** | ×3.80 / ×3.62 |
| **`P_ORACLE`** *(positive control)* | `iid` / `by_speed` | +48.3 % / +46.5 % ✅ better | **+54.6 % / +55.2 % ✅ better** | ×3.64 / ×3.89 |
| **`E0_v0`** *(negative bound)* | `iid` / `by_speed` | −74.2 % / −82.5 % ✅ worse | **−54.3 % / −53.6 % ✅ worse** | ×3.83 / ×3.54 |
| **`CV`** *(negative bound)* | `iid` / `by_speed` | −66.3 % / −71.2 % ✅ worse | **−44.8 % / −43.5 % ✅ worse** | ×3.77 / ×3.20 |
| **decorrelation control** *(the parent head's own along residual)* | `iid` / `by_speed` | −41.7 % / −43.1 % ✅ worse | **−23.4 % / −29.3 % ✅ worse** | — |

**Every control behaves. The run is not VOID.** The positive control stays separated-better, both
negative bounds stay separated-worse, and the decorrelation control stays separated-worse — so the
injection remains **conservative**, not optimistic, at n = 600.

⭐ **Measured power, an independent replication of `MODEL_REGISTRY §1.2a`:** CI half-widths shrink
**×3.20 – ×3.89, median ×3.76** over **18 arm×resampler cells** — against the registry's measured
**×2.8–3.9** and the **×3.87** a pure n^−½ law predicts. *(The registry's factor was measured on
eight open-loop driving metrics; this replicates it on a completely different statistic.)*

### 6.3 The verdict is robust to the background — all three carriers CONFIRM

| background | `E1_ego\|iid` n = 40 → n = 600 | `E1_ego\|by_speed` n = 40 → n = 600 |
|---|---|---|
| ⭐ **`parent_resampled`** (primary, conservative) | +13.3 % ❌ → **+25.4 % ✅** | +10.9 % ❌ → **+26.2 % ✅** |
| `sel` (zero-fit) | +28.6 % ✅ → **+40.8 % ✅** [−0.1432, −0.1083] | +25.8 % ✅ → **+41.4 % ✅** |
| `reduced` (ridge, re-fit at each n) | +29.2 % ✅ → **+41.2 % ✅** [−0.1445, −0.1096] | +26.5 % ✅ → **+41.9 % ✅** |

**The point estimate rises at n = 600 in every background** — not because the head improved, but
because §6.1's `a0` degrades more on the larger deployment than the goal-informed pick does.

### 6.4 ⛔ THE REGISTERED PREDICATE IS NO LONGER DISCRIMINATING AT n = 600 — said plainly

**`E1_noise_hist` — an arm whose four history columns are Gaussian noise — ALSO separates at
n = 600 (+9.1 %, [−0.0456, −0.0102], "BETTER").** At 600 clusters the test has so much power that
*any* arm carrying real kinematics separates. ⇒ **"`E1_ego` is separated" no longer distinguishes
"speed history is the lever" from "any half-decent along-track head is a lever".**
*(This is class **C13 — a guard that cannot fail — found in my own pre-registered primary. It is
reported, not worked around.)*

⭐ **The discriminating statistic is the DIRECT paired contrast on the same windows** — measured
because §5's arms were registered in advance for exactly this:

| contrast (n = 600, primary background) | paired Δ `ade_0_2s` | |
|---|---:|---|
| **`E1_ego` vs `E1_nohist`** — what 1 s of speed/accel history buys | **−0.0503 [−0.0517, −0.0489]** | ✅ SEP |
| **`E1_ego` vs `E1_noise_hist`** — history vs **noise in the same columns**, capacity fixed | **−0.0504 [−0.0519, −0.0490]** | ✅ SEP |
| ⛔ **`E1_nohist` vs `E1_noise_hist`** — **MUST BE NULL** | **−0.0001 [−0.0006, +0.0004]** | ⭐ **null** ✅ |
| `E1_ego` vs `E1_L` — the whole lead-vehicle block | **+0.0064 [+0.0056, +0.0071]** | ✅ SEP (lead better) |
| `E1_ego` vs `E0_v0` — kinematics+history vs `v0` alone | **−0.2457 [−0.2486, −0.2428]** | ✅ SEP |

> ⭐ **THE MECHANISM SURVIVES n = 600 ON A DIRECT CONTRAST, NOT ON A SEPARATION PREDICATE.**
> Of `E1_ego`'s **+25.4 %** recovery, the no-history head supplies **+9.1 %** ⇒
> ⭐ **16.3 of 25.4 points — 64.2 % of the recovery — is one second of speed/acceleration history.**
> And **fake history is statistically indistinguishable from no history** (Δ −0.0001, CI width
> 0.0010 at n = 600), so the 16.3 points are **information, not fitting slack.**

> ⛔ **The lead-vehicle block REPLICATES E-GOAL-1's refutation and now separates at its true size:**
> +0.0064 m, **+2.1 recovery points** (+25.4 % → +27.5 %) against speed history's **+16.3** —
> ⭐ **7.9× smaller.** *(E-GOAL-1 measured 4.4× on the head-fit RMS axis; on the recovery axis at
> n = 600 the gap is wider, not narrower.)* **Not re-scoped: agent tracks are not the lever.**

### 6.4a ⭐ Each rule's FAILING value was demonstrated, not asserted

`PRE_REGISTRATION §2.1` registered what input would make each verdict fire and claimed each was
reachable. **All three were reached in this run's own data, so no verdict was structurally
unavailable:**

| rule | what would make it fire | **demonstrated?** |
|---|---|---|
| **REFUTE** | the primary arm not separated-better | ⭐ **YES — the primary carrier returns exactly that at n = 40** (`E1_ego\|iid` −0.0358 [−0.1028, +0.0311], not separated). **REFUTE was live until the 600-episode fan finished decoding.** |
| separated-**WORSE** | an error structure that harms the pick | ⭐ **YES at n = 600** — `E0_v0` +0.1674 [+0.1494, +0.1856], `CV` +0.1380, decorrelation control +0.0721. |
| a genuine **NULL** at n = 600 | two arms that carry the same information | ⭐ **YES** — `E1_nohist` vs `E1_noise_hist` −0.0001 [−0.0006, +0.0004], with a CI **width of 0.0010** — i.e. the null is *tight*, not merely unpowered. |
| **CONFIRM** | a real, usable along-track signal | ✅ `P_ORACLE` +54.6 % separated-better at both n. |

### 6.5 ⚠️ MAE and RMS — which axis carries the verdict

**Stated in `PRE_REGISTRATION §6` before the fact, and it holds:** the `n` this stream raises is the
**placement episode count**. The **head-fit** RMS/MAE axis is a property of the frozen 612-clip dev
fit and is **NOT re-powered by anything here** — E-GOAL-1's finding that *at n = 612 clips the
along-track RMS axis cannot separate even a near-perfect oracle* stands untouched and is quoted as
`INHERITED`.

**On the placement axis both families are reported and they AGREE** (primary background, `iid`):

| arm | **MAE-family** (mean `ade_0_2s`) — the verdict axis | **RMS-family** (rms of per-window ade) |
|---|---|---|
| `E1_ego` | **0.4232 [0.4181, 0.4285]** | 0.4535 [0.4479, 0.4594] |
| `E1_nohist` | 0.4735 [0.4683, 0.4789] | 0.5026 [0.4970, 0.5084] |
| ⛔ `E1_noise_hist` | 0.4736 [0.4684, 0.4789] | 0.5028 [0.4972, 0.5086] |
| `P_ORACLE` | 0.3331 [0.3270, 0.3394] | 0.3771 [0.3705, 0.3843] |
| `E0_v0` | 0.6689 [0.6640, 0.6736] | 0.6949 [0.6898, 0.7000] |

⇒ **The verdict is carried by the MAE-family axis** (`ade_0_2s` is a mean-over-waypoints of L2, and
it is the axis every committed bar in this program is defined on). **The RMS-family companion gives
the identical ordering and identical conclusions at n = 600** — including the noise arm sitting on
the no-history arm. **No claim here rests on a disagreement between the two.**

### 6.6 ⭐ The requirement curve, on the family that applies — and it moved AGAIN

Family: **`EMPIRICAL`** — this stream's own frozen OOF along-track residuals, scaled by `k` and run
**through the real REF-C rule**. **Not `ISO`** (heavy-tailed, RMS/MAE ≈ 1.87 vs a Gaussian's 1.2533)
and **not `SHRINK`** (near-unbiased, α ≈ 0.996). Curve monotone ✅ in every cell.

| | σ₀ break-even | σ₅₀ half prize |
|---|---:|---:|
| inherited **`ISO`** bar (`GOAL_INPUT.md §5`) | 0.813 m | 0.439 m |
| E-GOAL-1, n = 40, `parent` background | 1.1434 m | 0.5907 m |
| ⭐ **this stream, n = 600, `parent_resampled`** | **1.2195 m** | **0.5851 m** |
| this stream, n = 600, `sel` / `reduced` | 1.3856 / 1.3912 m | 0.7957 / 0.8008 m |

> ⭐ **The measured head's along-track RMS is 0.9305 m. It CLEARS the n = 600 family-matched
> break-even by 1.31× — and FAILS the inherited `ISO` bar by 1.14×.** The family-matched σ₀ is
> **1.50× more forgiving** than the `ISO` number. ⇒ **reading the `ISO` bar literally still returns
> REFUTE where running the rule returns +25.4 %, separated** — class
> `BAR-INHERITED-FROM-THE-WRONG-FAMILY`, firing again at n = 600.
> ⚠️ **σ₀ is itself background-dependent (1.22 → 1.39 m) and deployment-dependent** — so
> **`BAR-INHERITED-FROM-THE-WRONG-FAMILY` needs a second clause: a requirement expressed as a scalar
> RMS needs its family, its deployment AND its background.**
> ⚠️ The curve still goes hard negative — **−298.7 % at 4.49 m** — so a goal channel must be **gated
> on measured accuracy** before it is wired in. A confidently wrong goal is destructive, not neutral.

---

## 7. WHAT THIS LICENSES, AND WHAT IT DOES NOT

### 7.1 Settled

1. ⭐⭐ **CONFIRM, on the pre-registered rule.** At n = 600 `E1_ego` is **separated-better on BOTH
   resamplers** under the conservative primary carrier (+25.4 % / +26.2 %) **and under both secondary
   backgrounds**. The `by_speed` non-separation that made E-GOAL-1 PROVISIONAL was **UNPOWERED, NOT
   REFUTED**, and 600 episodes resolves it.
2. ⭐ **64 % of the recovery is speed history**, on a direct paired contrast (−0.0503 [−0.0517,
   −0.0489]) that a pure-noise version of the same columns does not reproduce (null, −0.0001).
3. ⛔ **The lead-vehicle refutation replicates and sharpens**: +2.1 recovery points vs history's
   +16.3 — **7.9× smaller**. **Not re-scoped.**
4. ⛔ **0 of 600 val episodes overlap the parity train corpus**, by pose fingerprint, on the copy the
   dump actually reads, verified across two hosts. Parity holds.
5. ⭐ **CI half-widths shrink ×3.20–3.89 (median ×3.76)** going 40 → 600 — an independent
   replication of `MODEL_REGISTRY §1.2a` on a different statistic.
6. ⚠️ **The recovery is conditional on the cross-track background** (+13.3 % … +29.2 % at fixed n),
   and E-GOAL-1 did not report that dependence. **Every future recovery number must name its
   background.**
7. ⚠️ **The 600-episode deployment is NOT uniformly easier**: REF-C-XL's `a0` **degrades** 0.4712 →
   0.5015 where v1's flagship improves.
8. 🟠 **The committed `fan_refc-xl-30k.pt` is not bit-reproducible** on a current pod (1/881 picks
   differ, Δ`a0` −0.0002, not separated) — quantified in §2.3.

### 7.2 What I refuse to conclude

- **NOT that `+25.4 %` is a deployable number.** The placement still injects a **resampled residual**,
  not a real head's per-window predictions, and **no head has been trained on our latent, on the
  canonical windows, or end-to-end.** E-GOAL-1's own threat #2 is untouched by raising `n`.
  ⇒ **E-GOAL-3 remains the decisive experiment**, and it can still fail.
- **NOT that the separation predicate proves the mechanism.** §6.4 shows it does not at n = 600.
  The mechanism claim rests on the direct contrast and on the null of the noise arm.
- **NOT that σ₀ = 1.2195 m is *the* bar.** It is the bar **for this error family, on the 600-episode
  deployment, under the `parent_resampled` background.** It moves to 1.39 m under the others.
- **NOT anything past 2 s**, and every number is a displacement/ADE number — blind to collision.
- **NOT that the head-fit RMS axis is now powered.** It is frozen at n = 612 clips and unchanged.

### 7.3 ⭐ Does the speed-history feature enter v5? **YES.**

`dv_0p5`, `dv_1p0`, `v_lag_0p5`, `v_lag_1p0` — **1 second of ego speed history, from IMU/CAN, zero
new sensors, zero new corpus** — enter the v5 selector's goal input. **The size v5 should be sized
against is the CONSERVATIVE one: `+25.4 % [−0.0960, −0.0606]` of the fan's headroom** on the
`parent_resampled` carrier, **not** the +40.8 % the more favourable backgrounds return. **Plan with
+25 %, and gate the goal channel on measured accuracy (§6.6) before wiring it in.**

---

## 8. ESCALATIONS — these must not sit in a file

1. 🔴 **`V5_PLAN.md §8` and `Gates/flagship-v5-retrain.PREP.md` item 6 must be updated by their
   owner: E-GOAL-1's PROVISIONAL tier on the +23.6 % is now **CONFIRMED at n = 600**, and the number
   to plan with is **+25.4 % under a named background**, not +23.6 % bare. **The speed-history
   feature is cleared to enter the v5 selector.**
2. 🔴 **Every recovery number in the program must now carry its cross-track background.** §4.2
   measures a **15.9-point** swing and a separation flip at fixed `n`. `V5_PLAN.md §8`'s
   *"+23.6 % from ego kinematics alone (separated)"* is **true only for the background E-GOAL-1
   used and did not name.**
3. 🟠 **`MODEL_REGISTRY §1.2a` should gain a REF-C-XL row.** `a0` **0.4712 (40) → 0.5015 (600)** is a
   counter-example to reading the 600 build as "easier" — the CV floor and v1 improve while REF-C
   degrades. The registry's *never mix deployments* rule is strengthened by it.
4. 🟠 **Footnote the committed `fan_refc-xl-30k.pt`** with §2.3: reproducible to Δ`a0` −0.0002 (not
   separated), **not** bit-exactly, and its `nav_mode` provenance reads `None` where a re-dump reads
   `follow_constant`. ⭐ **This measurement is the first evidence that the two are equivalent**, which
   `refc_rerank.py` had only asserted.
5. 🟠 **For `RETRACTION_LOG.md` — root-cause classes:**
   - ⭐ **`RECOVERY-CONDITIONAL-ON-AN-UNREPORTED-BACKGROUND`** — a headline improvement measured while
     a *second, unreported* learned quantity is held fixed inherits that quantity's error structure.
     Measured here at **15.9 recovery points and a separation flip**, at fixed `n`, fixed treatment.
     **The check is: vary the held-fixed background and report the span.** *(Sibling of
     `MARGINAL-MISTAKEN-FOR-CONDITIONAL`.)*
   - ⭐ **`PREDICATE-STOPS-DISCRIMINATING-AT-HIGH-n`** — the mirror image of
     `SEPARATION-CLAIMED-ON-AN-UNPOWERED-AXIS`. At n = 600 an **information-free control arm
     separates**, so "separated" alone no longer supports a mechanism claim. **A power upgrade must
     re-check that its own decision predicate still has a failing case at the new `n` — and if it
     does not, name the contrast that does.**
   - **`ABSENCE-FOUND-AT-ONE-LOCATION`** — fired twice here: the REF-C-XL checkpoint was in
     `/workspace/models`, not `/root/models` (a 3.02 GB transfer avoided), and a sibling's leak audit
     covered a *different copy on the same pod* than the one the dump reads.
   - **`FIDELITY-GATE-TOO-STRICT-BY-ONE-VARIABLE`** — F-A compared across `n` **and** across
     host/vintage at once, so its failure was uninterpretable. **The fix is a 57-second experiment
     that varies one at a time**, not a weakened gate: pod2-40 vs pod2-600 is **bit-identical**.

---

## 9. THREATS TO VALIDITY I COULD NOT REMOVE

| threat | direction | status |
|---|---|---|
| ⛔ **The placement injects a RESAMPLED residual, not a real head's per-window predictions.** | measured **conservative** | the decorrelation control is separated-**worse** at both n (−23.4 % at 600) ⇒ every recovery is if anything an **under**-estimate. **But it is not a trained head, and raising `n` does not change that.** |
| ⛔ **E-GOAL-1's exact cross-track background cannot be built at n = 600** (`eh2_cache.pt` is 881-window-only; rebuilding it needs a second model and a second GPU pass). | **unknown, and large** | §4.2 measures the span (15.9 points). The primary carrier reproduces the parent's cross **error structure** (MAE 0.400 m) but **decorrelated**, which §4.2 shows is the conservative end. **Building `eh2` at 600 is the clean fix and is NOT done here.** |
| ⚠️ At n = 600 the separation predicate fires for an information-free arm | inflates confidence in "separated" | §6.4 — reported, and the discriminating contrast is measured instead. |
| The along-track pools are the **frozen 612-clip dev-corpus** fit; only the placement `n` rises | narrows what "n = 600" means | stated in `PRE_REGISTRATION §1` and §6.5. The head-fit RMS axis is **not** re-powered. |
| The dev-box corpus is keyed `14231cd29c74`, **not** parity `e438721ae894` | window set differs from canonical | E-GOAL-1's threat, inherited. **No parity-dependent step ran on the dev box**; the fan, the leak check and the val build are all pod2/parity. |
| `obstacle.offline` is a **label**, not perception | favours the lead hypothesis | every lead number is an **upper bound** — and the lead block is refuted anyway. |
| 2 s, displacement/ADE only | unknown, possibly large | §7.2. |

**Evidence classes.** §§1–6 are `MEASURED (ours)` with artifact paths. E-GOAL-1's +23.6 %/+20.3 %,
the 0.813/0.439 m `ISO` bars and the 1.151 m parent-head reference are `INHERITED` and are quoted
from raw JSON (`eg_place.json`, `gi_head_preds.npz`), **re-derived here and reproduced to 0.010
recovery points**. `MODEL_REGISTRY §1.2a`'s ×2.8–3.9 is `PUBLISHED` and is **independently
replicated** (§6.2). §7.2's E-GOAL-3 is `HYPOTHESIS` with a named test.

> **TIER: CONFIRMED** — for *"an along-track goal with this measured error structure recovers a
> separated, positive share of the REF-C fan's headroom at n = 600, on both resamplers and all three
> backgrounds, and ~64 % of it is ego speed history."*
> **STILL NOT DEPLOYABLE** — for *"a trained goal head will deliver +25 %."* That needs **E-GOAL-3**
> (the head fitted on `physicalai-train-e438721ae894` and scored on the canonical windows with its
> **actual, correlated** per-window predictions), and it can fail.

---

## 10. DELIVERABLE MANIFEST

**Everything below is staged in the repo working tree (`git add`). Nothing was committed or pushed.**
⚠️ **Exactly two artifacts exist in only one place, and both are named here rather than left to an
audit.**

| artifact | where | what |
|---|---|---|
| `EGOAL_2.md` | `repo:…/incoming/2026-07-28-egoal-2-power/` | this document |
| `PRE_REGISTRATION.md` | same | bars, arms, failing-value proofs, **the bridge condition that fired** — written before any fit; §9 amendments appended, nothing deleted |
| `code/e2_leak.py` | same | S0 — pose-sha256 fingerprint, train overlap, prefix, cross-copy identity |
| `code/e2_dump600.py` | same | S1 — the 600-episode fan; calls the **committed** `refc_rerank.dump()` unmodified |
| `code/e2_fanmatch.py` | same | gate F-A — per-field prefix identity, the gate that failed and then isolated its cause |
| `code/e2_arms.py` | same | the both-directions arms (`E1_nohist`, ⛔ `E1_noise_hist`) + the bit-identity fidelity control |
| `code/e2_place.py` | same | ⭐ the placement; **imports** E-GOAL-1's `realise`/`pick_nearest_to`/`goal_reference`; 4 cross-track backgrounds; mechanism contrasts; family-matched curve |
| `code/e2_summary.py` | same | the within-construction n = 40 → n = 600 table + the mechanical verdict |
| `raw/e2_leak.json` | same | ⭐ the leak check — 0/600, with all 600+600 per-episode fingerprints |
| `raw/e2_pub40_fingerprint.json` | same | the eval pod's own 40-episode fingerprints (the two-host half of check B) |
| `raw/e2_fanmatch.json` · `…_pod2_40_vs_600.json` · `…_committed_vs_pod2_40.json` | same | F-A and its discriminating experiment |
| `raw/e2_arms.json` · `raw/e2_extra_pools.npz` | same | the extra arms + their pools (so every §6.4 contrast recomputes on any CPU) |
| `raw/e2_place_n40_{parent,reduced,sel,parentresamp}.json` | same | the **committed-fan** n = 40 cells — F-B and the background-sensitivity table (§4.2) |
| `raw/e2_place_n40p2_{parent_resampled,sel,reduced}.json` | same | the **pod2-fan** n = 40 cells — the host-matched baselines for §6 |
| `raw/e2_place_n600_{parent_resampled,sel,reduced}.json` | same | ⭐ the n = 600 placements |
| `raw/e2_summary.json` | same | the paired table, the measured power factors, the verdict |
| ⚠️ `fan_refc-xl-30k_600ep.pt` (123 MB) | **`pod2:/workspace/_egoal2/` + dev-box scratchpad ONLY** | the 600-episode fan. `.gitignore` bans large binaries repo-wide; **not `git add -f`ed** — see below |
| ⚠️ `fan_refc-xl-30k_40ep_pod2.pt` (8.2 MB) | **same two places** | the host-matched 40-episode fan |

### ⚠️ The two artifacts in only one place — stated, not discovered later

Neither fan is staged. **Why this strands nothing:**
- **The producer is staged** (`code/e2_dump600.py`) and it calls the **committed** decode; the dump
  regenerates in **19 min of A40** (600 eps) / **57 s** (40 eps) on pod2, and the checkpoint is
  HF-backed (`Sayood/tanitad-refc-xl`, md5 verified) so it is not single-disk.
- **Every number in this document is recomputable** from the staged `raw/*.json` + `raw/*.npz` +
  `taniteval/ci.py`, without the fans.
- **They also live on pod2**, so they are not on a single disk.
- **Decision needed from the owner:** if a 123 MB fan should be shareable, the right home is HF under
  `Sayood/` (gated), not a `git add -f` against a repo-wide policy. **Flagged rather than actioned.**

**Inputs consumed** (read-only): `pod2:/root/valdata/physicalai-val-0c5f7dac3b11` (600 eps),
`pod2:/workspace/data/physicalai_phase0/_epcache/physicalai-{train-e438721ae894,val-0c5f7dac3b11}`,
`pod2:/workspace/models/refc-xl-30k/`, `eval:/root/valdata/physicalai-val-0c5f7dac3b11` (40 eps),
`taniteval/results/fan_refc-xl-30k.pt`, `…/2026-07-27-egoal-1-lead-vehicle/raw/{eg_oof_pred_gbm.npz,
eg_windows.parquet,eg_place.json}`, `…/2026-07-27-goal-input/raw/gi_head_preds.npz`.

⛔ **Parity untouched:** `_epcache` never written, no episode re-selected; the parity train corpus was
**read only to fingerprint it** for the leak check.
🔒 No clip UUID or raw PhysicalAI content reaches any artifact.

**Suite green:** `cd stack && pytest -q` → **1259 passed, 12 skipped** in 82 s (2026-07-27, after this
stream). This stream added **no files** to `stack/` or `taniteval/`; all new code lives in the hub
folder and **imports** the repo harness rather than modifying it.
