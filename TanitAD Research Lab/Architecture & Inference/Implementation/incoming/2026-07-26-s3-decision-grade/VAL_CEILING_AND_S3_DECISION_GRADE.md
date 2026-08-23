# The val arithmetic ceiling, and S3 on the parity corpus

**Date:** 2026-07-26 (Europe/Berlin) · **Stream:** 4-Brain Dominance Program
**Tasks:** (1) the n ≥ 200 arithmetic ceiling that gates the whole HP-1…HP-8 ladder · (2) S3's
decision-grade re-run on the **parity** caches · (3) **S3-W** beside it
**Compute:** **pod3** (A40, 0 MiB at start — free). pod1 untouched-for-compute, **pod2 read-only**,
eval pod read-only. No GPU used by this work. No training launched.
**Nothing `git add`-ed, committed or pushed.**

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another of our docs, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline

| # | Result | Class |
|:--:|---|---|
| **1** | ✅ **The 600-episode CLEAN val build EXISTS.** `physicalai-val-0c5f7dac3b11`, **exactly 600 `ep_*.pt`**, 0 skips, 66 GB — on **pod2**, `/workspace/data/physicalai_phase0/_epcache/`. | `MEASURED` |
| **2** | ✅ **It is byte-level episode-disjoint from the parity train.** **0 / 600** episodes share a `sha256(poses)` with any of the 2,376 parity-train episodes. | `MEASURED` |
| **3** | ⭐ **The published 40-episode eval deployment is the exact first-40 PREFIX of the 600 build** (and pod1's 12 is the first-12). Extending to 600 **adds** episodes and **re-selects none** — parity is preserved, not broken. | `MEASURED` |
| **4** | ⛔ **`episode_id` is a COLLIDING key and must never carry a disjointness verdict.** The parity train has 2,376 episodes but only **2,342 unique `episode_id`s**. It over-reports overlap; the 600-ep val shows **3.3 %** overlap on `episode_id` and **0.0 %** on bytes. | `MEASURED` |
| **5** | ✅ **YES — the ladder's n ≥ 200 two-arm comparisons CAN be run**, on the 600-episode build. **NOT on the eval pod's 40** (hard ceiling), and the eval pod **physically cannot host** the 600 build today: its `/` is **99 % full, 2.9 GB free**, against a 66 GB requirement. | `MEASURED` |
| **6** | ⚠️ **"the 600-ep val is on the training pods" (plural) is wrong.** It is on **exactly one host — pod2**. pod1's `_epcache` root holds **no val dir at all**; pod3 holds **no clean val at any path**. The one pod that has it is the one pod that is off-limits. | `MEASURED` |
| **7** | ✅ **VAL_PARITY_REPORT §4.3/§7.1's open blast-radius question is CLOSED: NO root co-locates both val splits.** `LEADERBOARD.md §8` does **not** need re-derivation on those grounds. | `MEASURED` |
| **8** | ⭐ **S3 re-run at parity: the power verdict HOLDS (558 lat / 520 lon clusters ≥ 200), but the characterisation does NOT.** Four numbers moved materially. | `MEASURED` |
| **9** | ⛔ **S3's "the highway stratum does not exist" is OVERTURNED.** 2 clusters on the dev-box sample → **229 at parity**, the *largest* stratum by windows. The city/mid-only scoping of HP-2 was a sampling artifact. | `MEASURED` |
| **10** | ⛔ **The skill bars nearly DOUBLED. S3: 0.3898 → 0.6534 (lat), 0.2334 → 0.5323 (lon). S3-W: 0.1128 → 0.2566, 0.0676 → 0.2881.** Quoting the brief's bars against a parity-corpus arm would credit a **−0.10 loss as a +0.16 win**. | `MEASURED` |
| **11** | ⛔ **R2 (echo) is now ARMED ON BOTH AXES** — the longitudinal `B3 − B1` moved from *not separated* to **+0.2442 [+0.1966, +0.2951] separated**. **S3-W is the variant that measures anything.** | `MEASURED` |

---

## 1. ⛔ TASK 1 — the ceiling

### 1.1 Method: count episodes and hash bytes, never trust a directory name

The standing finding is that *"the val corpus is deployed at 600 / 40 / 12 episodes under the **SAME**
directory name"*. A name is therefore not evidence, and neither is a count on its own. Every row below
is keyed on **`sha256` of the raw `poses[T,4]` float32 bytes**.

**Why that key.** Each `ep_*.pt` is ~117 MB, dominated by `frames_u8 [T,9,256,256]`; `poses` is ~3 KB.
Reading via `torch.load(..., mmap=True)` faults in **only the poses pages**, so hashing a 2,376-episode
cache costs **121 s and a few MB of IO** instead of 260 GB. That is what made a byte-level check of
every deployment affordable, including on pods I must not load.

⛔ **`episode_id` was tried first and REJECTED.** `MEASURED`: the parity train holds 2,376 episodes and
only **2,342 unique `episode_id`s** — 34 collisions; the 600-ep val holds 600 with **596** unique.
Sample values (`808464740`, `808465461`, …) decode as a **4-character prefix of the clip id packed as
int32**, so the key is lossy by construction. It **over-reports** overlap in every direction:

| pair | overlap by `episode_id` | overlap by **bytes** |
|---|---:|---:|
| clean val @600 vs parity train | 20 / 600 = 3.3 % | **0 / 600 = 0.0 %** |
| S3's local train-14231cd29c74 @400 vs parity | 70 / 400 = 17.5 % | **58 / 400 = 14.5 %** |
| S3's local val-bb543bdf7836 @100 vs parity | 17 / 100 = 17.0 % | **14 / 100 = 14.0 %** |
| leaked val-f1b378f295ae @80 vs parity | 62 / 80 = 77.5 % | **62 / 80 = 77.5 %** |

The 20 apparent "overlaps" of the clean val are **key collisions, not leakage**.

### 1.2 ⭐ THE INVENTORY — every val/train cache, per pod, per root, by episode count

`MEASURED` 2026-07-26. Full record: `val_corpus_inventory.json` (this directory).

| host | path | cache | **eps** | byte-overlap w/ parity train | clean? |
|---|---|---|---:|---:|:--:|
| **pod3** | `/workspace/pai_epcache/` | `physicalai-train-e438721ae894` | **2376** | (is the parity train) | — |
| ⭐ **pod2** | `/workspace/data/physicalai_phase0/_epcache/` | `physicalai-val-0c5f7dac3b11` | **600** | **0** (0.0 %) | ✅ **CLEAN** |
| **eval pod** | `/root/valdata/` | `physicalai-val-0c5f7dac3b11` | **40** | **0** (0.0 %) | ✅ CLEAN — **the published set** |
| **pod1** | `/root/valdata/` | `physicalai-val-0c5f7dac3b11` | **12** | **0** (0.0 %) | ✅ CLEAN (partial) |
| **pod3** | `/workspace/v4run/valcache/` | `physicalai-val-heldout-79d4e3d2d4c6` | **44** | **0** (0.0 %) | ✅ CLEAN — all 44 ⊂ the 600 |
| ⛔ **pod3** | `/workspace/pai_epcache/` | `physicalai-val-f1b378f295ae` | **80** | **62** (77.5 %) | ⛔ **LEAKY — REFUSED** |
| pod3 | `/workspace/pai_epcache/` | `physicalai-train-51f40f5ebc21` | 320 | 256 (80.0 %) | older train build |
| dev box | `…\tanitad-data\physicalai\_epcache\` | `physicalai-train-14231cd29c74` | 400 | 58 (14.5 %) | **non-parity** (S3's first pass) |
| dev box | `…\tanitad-data\physicalai\_epcache\` | `physicalai-val-bb543bdf7836` | 100 | 14 (14.0 %) | **non-parity** (S3's first pass) |

**Absence probed at more than one location, per `CLAUDE.md`.** pod3 was enumerated by recursive `find`
to depth 5 over `/workspace /root /data /mnt` **plus** an `ep_*.pt` count for every directory found.
pod1, pod2 and the eval pod were probed with targeted read-only `ls` over nine candidate roots **and** a
shallow glob for `*val*`, `*0c5f7dac3b11*` and `*f1b378f295ae*`. **No other PhysicalAI val epcache
exists on any of them.** (pod2 also carries `/workspace/data/ijepa_feats_smoke/physicalai-val-0c5f7dac3b11-ijepa-vith14`
— a *feature* directory, not an epcache, no `ep_*.pt`.)

### 1.3 Is the 600-episode build CLEAN? — verified at the byte level, not assumed

**✅ YES. 0 / 600.** No episode of `physicalai-val-0c5f7dac3b11 @600` shares a `sha256(poses)` with any
of the 2,376 episodes of `physicalai-train-e438721ae894`.

Two independent corroborations, neither of which I relied on alone:

* `MEASURED`, pod2 `/workspace/v15/labels_val_v4_provenance.json`: `n_episodes: 600`,
  `n_windows: 102532`, `parity_key: physicalai-train-e438721ae894`,
  `skip_hash: f09e44db (unchanged — labels re-derived on the existing pose cache; **no episode
  re-selection**)`. It also carries `v21_parity: {bit_identical: true}`.
* `MEASURED`, this run: the 600 build's clip-length range is `T ∈ [188, 205]` — **identical** to the
  parity train's `[188, 205]`, i.e. the same builder on the same R0 corpus.

⭐ **And the published set sits INSIDE it.** The eval pod's 40 episodes are the **exact, order-preserving
first-40 prefix** of the 600 (all 40 hashes match `m600[:40]` in order); pod1's 12 are the first-12.
**This is the fact that makes the ceiling fixable without breaking parity**: moving from 40 → 600 is a
strict superset, not a re-selection, so `Parity is sacred` is not violated.

⚠️ **Numbers on 600 episodes are still not directly comparable to numbers on 40** — different n, different
windows. The corpus is consistent; the *statistics* must be re-run, not rescaled.

**The clean deployments nest exactly** (`MEASURED`, byte level):
`pod1's 12` ⊂ `the published 40` ⊂ `the 600 build`.

⚠️ **One thing that does NOT nest, flagged for its owner.**
`physicalai-val-heldout-79d4e3d2d4c6` (44 eps, pod3 `/workspace/v4run/valcache`) is a clean subset of the
600 (0 parity-train leakage, all 44 ⊂ the 600) — **but it shares 3 episodes with the published
40-episode eval set**. It is therefore *not* held out from the published set, and a number quoted on it
is not independent of the published deployment. Small, but it is exactly the class of thing that gets
assumed rather than measured. **Owner: whoever uses `v4run` / E1c.**

### 1.4 ⛔ THE ANSWER — can the ladder's n ≥ 200 two-arm comparisons be run, and on what?

> ### ✅ **YES — but on the 600-episode build, which today exists on exactly one pod, and that pod is pod2.**

**S3's ⛔ verdict on the 40-episode set was an `ESTIMATED` projection (32 lat / 29 lon). It is now
`MEASURED` on the real published episodes** — they are the first-40 prefix of the 600 view, so this is
the actual published set, not a proxy (`s3_power_MEASURED_published_subsets.json`):

| deployment | episodes | **lat clusters** | **lon clusters** | lat *with an event* | ≥40 single-arm | ≥200 two-arm |
|---|---:|---:|---:|---:|:--:|:--:|
| pod1 `/root/valdata` | 12 | **11** | **11** | 4 | ⛔ | ⛔ |
| ⛔ **eval pod `/root/valdata` — every published number** | **40** | **37** | **34** | **14** | ⛔ **below 40** | ⛔ |
| pod3 `v4run/valcache` heldout | 44 | 41 | 37 | 15 | ~ lat only | ⛔ |
| ⭐ **pod2 `…/_epcache/physicalai-val-0c5f7dac3b11`** | **600** | **see §2.4** | **see §2.4** | — | ✅ | ✅ |

> ⛔ **The published 40-episode eval set fails BOTH bars, measured: 37 lat / 34 lon clusters — under the
> 40-cluster SINGLE-arm bar, not merely the 200 two-arm bar.** S3's projection (32/29) was slightly
> pessimistic in magnitude and **exactly right in verdict**.
>
> ⚠️ **Worse than the cluster count suggests: only 14 of the 40 episodes carry a lateral event at all**
> (18 of 40 longitudinal). A five-class ordinal statistic resting on 14 event-carrying episodes is not a
> decision-grade instrument, whatever its CI says.

**The ceiling is real and it is not an S3 problem.** 40 episodes cannot yield 200 episode-clusters at any
yield ≤ 1. **Every HP-1…HP-8 prediction whose power requirement is "n ≥ 200 clusters" is blocked on the
eval pod's deployment**, regardless of the problem's own yield. No amount of window-level data fixes it,
because the resampling unit is the episode.

**What "runnable" costs, split by what the problem needs:**

| problem class | needs | status |
|---|---|---|
| ✅ **label-only / poses-only** (S3, option-set and coverage characterisation, blind firewalls) | `poses[T,4]` = **4.8 MB for all 600** | ✅ **runnable NOW.** Done in this work: a poses-only view of all 600 val + all 2,376 train episodes is on pod3. |
| ⚠️ **two-arm MODEL comparisons** (the actual ladder) | the full 66 GB cache **with `frames_u8`** on an eval-capable pod | ⚠️ **blocked on one file move**, quantified below |

**The operational cost of unblocking the model comparisons — `MEASURED` inputs, `ESTIMATED` total:**

* The 600-ep cache is **66 GB** (`du -sh`, pod2). It exists **only** on pod2.
* ⛔ **The eval pod cannot receive it at `/root/valdata` as deployed**: its `/` overlay is **200 G total,
  198 G used, 2.9 G free (99 %)**, and `/root/valdata` lives there. 66 GB does not fit. `MEASURED`.
  ⚠️ Its `/workspace` (MooseFS) is a *possible* alternative target but I did **not** probe its quota —
  that needs a multi-GB `dd` write and the eval pod was running a sibling agent's job at 99 % GPU
  throughout. **UNDETERMINED**, and the command that settles it is
  `dd if=/dev/zero of=/workspace/_ddt/t.bin bs=1M count=8000` on an idle eval pod (never `df` — it
  reports the cluster, not the quota). Note that every evaluator's val path is currently
  `/root/valdata/...`, so using `/workspace` is also a config change, not only a copy.
* ✅ **pod3 can**: `/` overlay is **600 G total, 36 G used, 565 G free**. `MEASURED`. (pod3's
  `/workspace` MooseFS is the *wrong* target — `du` already accounts ~483 GB there against a per-pod
  quota the program measures at ~466 GB. An 8 GB `dd` write succeeded at 122 MB/s, so there is *some*
  headroom, but not 66 GB of it. Judge by `dd`, never `df` — `df` reports the 965 TB cluster.)
* **Path:** pods cannot SSH each other and the dev-box relay is ~1 MB/s (66 GB ≈ **18 h — unusable**).
  The program's measured fast path is **push → HF from the source pod at ~118 MB/s, then pull**:
  ≈ 10 min up + 10 min down + md5 verify ≈ **30–60 min wall clock**, `ESTIMATED` from the measured
  118 MB/s figure.
* ⚠️ **The source pod is pod2, which is mid-run and RAM-bound.** The push should wait for its 30 k run
  to finish rather than contend with it. **This is a scheduling decision for the PI, not a technical
  blocker.**

**Per-stratum, the answer is narrower than the pooled one** — see §2.5 for the measured strata on the
real 600-episode val.

### 1.4.1 ⛔ The harder ceiling behind the first one: **600 is the maximum, ever, at parity**

`MEASURED`, `stack/scripts/parity_skipset.sh` (read directly, not cited from prose): the phase-0 corpus
is **3,000 R0 clips**; `split_clips` does `randperm(3000, seed 0)` and takes the **first 600 as val**,
leaving 2,400 train (2,376 after the 24 corrupt skips → skip-hash `f09e44db`).

> **Therefore the clean val can never exceed 600 episodes without re-selecting episodes — which
> `CLAUDE.md` forbids outright ("Parity is sacred… anything that re-selects episodes breaks cross-arm
> comparability and must be refused").**

That converts the ladder's power bar into a **single arithmetic test**, which is the useful form:

> ### A two-arm HP-x comparison is feasible at parity **iff its episode yield ≥ 200 / 600 = 0.333**.

| problem | measured episode yield | max clusters on the 600 | ≥200? |
|---|---:|---:|:--:|
| **S3 lateral** | **see §2.4** (measured this run) | — | see §2.4 |
| **S3 longitudinal** | **see §2.4** | — | see §2.4 |
| any problem with yield < 1/3 | < 0.333 | < 200 | ⛔ **infeasible at parity, at any deployment** |

**A problem below 1/3 yield cannot be fixed by moving files.** It needs either a different resampling
unit, a pooled multi-problem estimand, or an explicitly non-parity corpus — and the last one forfeits
comparability with every existing arm. **This should be checked per HP before compute is booked, not
after.**

**The non-parity escape hatch, for completeness.** `MEASURED`, pod3
`/workspace/data/physicalai_v2/r0/r0_selection.parquet`: the **v2** corpus carries **9,000 clips**
(3× phase-0), which at the same 20 % split would allow ~1,800 val episodes. ⚠️ **But any arm evaluated
there is not comparable to any existing arm** — it is a new parity key, not a bigger val for the current
ladder. It is a *next-corpus* answer, not a fix for HP-1…HP-8 as currently defined.

### 1.5 ✅ A second, unasked question this closed

`VAL_PARITY_REPORT.md` §4.3 / §7.1 flagged **the only open question that could still invalidate a
published table**: *does any single epcache root list **both** val split dirs?* If so, the
`sorted(glob("*val*"))[-1]` resolver — which lexicographically **prefers** the leaked split — would have
silently selected it, and `LEADERBOARD.md §8` would need re-derivation.

**`MEASURED`, all four pods, 2026-07-26: NO root co-locates the two splits.**

| root | holds |
|---|---|
| pod3 `/workspace/pai_epcache` | the **leaky** val only (+ two train builds). No clean val to lose. |
| pod2 `…/physicalai_phase0/_epcache` | the **clean** val only (+ the parity train). `[-1]` picks clean. |
| pod1 `…/physicalai_phase0/_epcache` | the parity train **only** — no val dir. |
| eval pod `/root/valdata` | the **clean** val @40 + comma + cosmos. No leaky split. |

⚠️ **Caveat, stated rather than buried:** this is the disk state **today**. It is not a statement about
disk state at the time of any historical run, and no committed artifact records that. The finding is
**LATENT, never FIRED, on the current state** — which is materially different from "it fired".

### 1.6 Corrections to inherited figures

| inherited claim | measured here | verdict |
|---|---|---|
| `f1b378f295ae` is **78.5 %** leaked | **62 of 80** files = **77.5 %**. The *same* 62 episodes give **62/79 = 78.48 %** on the 79 **unique `episode_id`** denominator. | **CONFIRMED in substance** — same 62 episodes, different denominator. Stays **REFUSED**. |
| S3's local caches overlap parity by **17.5 % / 17.0 %** | byte-level **14.5 % / 14.0 %** | S3's caveat and its conclusion (the caches are non-parity) **stand**; the magnitude was inflated by the colliding key. |
| the 600-ep val is "on the training pod**s**" | on **exactly one** host: **pod2** | ⚠️ **corrected.** pod1 has no val dir in `_epcache`; pod3 has no clean val anywhere. |

---

## 2. TASK 2 — S3 on the parity corpus

### 2.0 How the parity run was made possible without touching pod2's compute

The re-run needs the parity **train** (2,376 eps, on pod3 ✅) and the clean **val** (600 eps, **pod2
only**). pod2 is mid-run and off-limits, and the val cache is 66 GB.

**Resolution: a poses-only VIEW.** S3's miner reads **only** `ep["poses"]` ([T,4], ~3 KB); the 117 MB is
`frames_u8`. Reading with `torch.load(..., mmap=True)` faults in only the poses pages, so:

| step | cost | `MEASURED` |
|---|---|---|
| extract 600 val episodes on pod2 | **17 s**, ~5 MB written to `/tmp` (not MooseFS), mmap reads | ✅ |
| bundle + relay pod2 → dev box → pod3 | **1.86 MB**, md5 `d9ccf95e…` verified at **both** ends | ✅ |
| extract 2,376 parity-train episodes on pod3 | **121 s** | ✅ |

> ⭐ **The view is PROVEN equivalent, not assumed.** Mining 50 episodes from the **full 117 MB/episode
> parity cache** and from the **view** gives **8,540 rows each** and a **bit-identical `sha256`
> (`9a844918f7a57f3d…`) over every mined field**. The poses tensors themselves also carry per-episode
> `sha256` recorded on both sides. **This is a lighter view of the same corpus, not a different corpus.**

**`--cache-dir` swap only, exactly as pre-registered.** The four S3 files on pod3 are **md5-identical**
to the repo copies (`s3_labels.py c1b08d7b…`, `s3_blind_baseline.py ed0af06a…`,
`run_s3_characterisation.py 620a8273…`, `test_s3_labels.py 7122c718…`). **`pytest -q` → 20 passed** on
pod3, matching the dev box. The five label/CI modules S3 depends on
(`refb_labels.py`, `v4_labels.py`, `lake/vtarget.py`, `lake/vocab.py`, `taniteval/ci.py`) are **md5-identical
between pod3 and the repo** — checked because pods drift out of sync, and a stale label module would
silently change the target. The run prints **`PARITY=YES`** for both caches.

### 2.0.1 ⚠️ An operational trap this run walked into — worth more than the run itself

`MEASURED`, pod3: the first launch mined at **2.07 s/episode**. After capping BLAS/OMP threads it mined
at **0.223 s/episode — a 9× speed-up.** Cause: numpy/torch spawned **111 threads** for per-window
operations on arrays of a few hundred elements, where thread dispatch dominates the arithmetic.

⛔ **And launching a *second* such job took throughput to ~zero.** Two processes × 111 threads on 96
cores drove the load average to 34.75 and the primary's log **stopped advancing for 10 minutes** while
both processes burned ~380 % CPU each. **The stall began at 06:55:12, the second job launched at
06:54:59.** I caused it, and I only found it because the log timestamps lined up.

> **Standing advice for any CPU mining/labelling job on these pods:**
> `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`.
> `MEASURED`: mining is **flat at 0.22–0.23 s/ep for 1, 4 and 8 threads** and **9× worse at the 111-thread
> default**. This is not S3-specific — it applies to every per-window numpy labeller in the program.
> Cost here: ~25 wasted pod-minutes. Cost if it had gone unnoticed on a long run: hours.

### 2.1 ⭐ Headline: the characterisation does **NOT** hold. Four things moved materially.

The brief asked me to flag material moves rather than quietly adopt them. **Four moved, and two
overturn a stated S3 conclusion.** All `MEASURED`; full side-by-side in `s3_parity_vs_nonparity.json`.

| # | quantity | non-parity (400/100 dev-box) | **PARITY (2376/600)** | verdict |
|:--:|---|---:|---:|---|
| **1** | ⛔ **highway stratum**, lat clusters | **2** ("does not exist") | **229** | ⭐ **OVERTURNED** |
| **2** | lateral **majority-class rate** | 0.3255 (`t_5_10`) | **0.8553** (`t_none`) | ⭐ **option set changed character** |
| **3** | **S3 skill bar**, lateral | 0.3898 | **0.6534** | ⭐ **the bar nearly doubled** |
| **4** | longitudinal **echo** (B3−B1) | +0.0524, **not** separated | **+0.2442 [+0.1966, +0.2951] SEPARATED** | ⭐ **R2 now armed on BOTH axes** |

**Which is decision-grade, and why.** The **parity** numbers. Not because they are newer, but because
the non-parity numbers fail an independent consistency check:

> `MEASURED`: the shipped label artifact for the *same* 600-episode val (pod2
> `labels_val_v4_provenance.json`) records `strat_scalars.ttm = 0.2076` — the fraction of windows with a
> curvature segment in a **25 s** lookahead. S3's parity lateral event rate over a **12 s** horizon is
> **0.1447**, correctly **below** it. ✅
> The non-parity rate is **0.7178 — 3.5× the 25 s figure**, which is impossible for the same corpus.
> **The dev-box caches are a city-heavy sample, not a scaled-down parity corpus.**

S3's own §1.1 said transfer was "bounded, not assumed" and predicted *"coverage and class balance may
shift on parity; the firewall verdicts and the power arithmetic will not."* **Half right:** the power
arithmetic held (both bars still clear), but the **firewall verdicts did move** — the longitudinal echo
flipped from not-separated to separated, and every bar rose.

### 2.2 ⛔ The highway reversal — the single most consequential change

`MEASURED`, `s3_power_parity_primary.json` → `strata_val`:

| stratum | non-parity lat clusters | **PARITY lat** | **PARITY lon** | PARITY lat windows |
|---|---:|---:|---:|---:|
| city `< 8 m/s` | 65 | **235** ✅ | 178 ⛔ | 8,533 |
| mid `8–15 m/s` | 50 | **224** ✅ | 179 ⛔ | 10,387 |
| ⛔→⭐ **highway `≥ 15 m/s`** | **2** | **229** ✅ | **214** ✅ | **15,417 (the LARGEST)** |

> ⛔ **S3_IMPLEMENTATION.md §3.3 states: *"The highway stratum does not exist for S3 on this corpus — 2
> val clusters, 7 train clusters… Any S3 result is a city + mid-speed result and must be labelled as
> one."* On the parity corpus this is FALSE.** Highway carries **229 lateral clusters** — clearing the
> 200 two-arm bar — and **more windows than either other stratum**.
>
> ⚠️ **And the stratified feasibility INVERTS on the longitudinal axis:** city (178) and mid (179) now
> **fail** the 200 bar while **highway (214) passes** — the exact opposite of the non-parity reading.

**Consequence for HP-2.** HP-2 is a *stratified* prediction, and S3's non-parity result would have
scoped it to "city + mid-speed only". **That scoping was an artifact of the dev-box sample.** At parity,
the lateral axis supports all three strata at n ≥ 200, which is strictly better news for the ladder —
and it means a highway-vs-city interaction test is available where S3 said it was not.

### 2.3 Coverage and class balance — the option set changed character

`MEASURED`, val, `s3_coverage_parity_primary.json`:

| | lat non-parity | **lat PARITY** | lon non-parity | **lon PARITY** |
|---|---:|---:|---:|---:|
| all windows | 17,100 | **102,532** | 17,100 | **102,532** |
| M1 full 12 s observable | 7,200 (42.1 %) | **43,132 (42.1 %)** | — | — |
| M1∩M4 moving | 5,079 | **38,481** | 5,079 | **38,481** |
| ✅ admissible | 3,459 | **34,337** | 1,997 | **24,987** |
| coverage of all windows | 20.23 % | **33.49 %** | 11.68 % | **24.37 %** |
| **majority class** | `t_5_10` | ⭐ **`t_none`** | `t_2_5` | ⭐ **`t_none`** |
| **majority rate** | 0.3255 | ⛔ **0.8553** | 0.3655 | ⛔ **0.6146** |
| event rate | 0.7178 | **0.1447** | 0.8027 | **0.3854** |

Class balance, lateral: `t_1_2` 9.37→**1.67 %** · `t_2_5` 26.37→**5.18 %** · `t_5_10` 32.55→**6.80 %** ·
`t_10_H` 3.50→**0.81 %** · `t_none` 28.22→**85.53 %**.

> ⛔ **On the parity corpus the lateral problem is 85.5 % "no manoeuvre in the next 12 s".** The four
> event bands together hold 14.5 % of the mass, and the thinnest (`t_10_H`) holds **0.81 %**.
> **Per-band recall on `t_10_H` cannot carry any verdict at this balance** — S3's own §4.2 warning, now
> much sharper.
>
> ⚠️ **The pre-registered SECONDARY equal-mass banding no longer does its job.** It was chosen because
> equal-mass *minimises* the majority baseline. At parity its majority rate is **0.8553 (lat)** —
> **identical to the primary**, because `t_none` alone exceeds any quartile. The robustness arm has
> silently degenerated. **`MEASURED`; reported, not fixed — re-banding after seeing this would be exactly
> the post-hoc move the pre-registration forbids.**

**Bands and thresholds untouched**, per the pre-registration: edges remain the spec's **2 / 5 / 10 s**;
`A_MAN = 0.5`, `DV_MIN = 1.5`, `MIN_TTM_S = 1.0`, `H_S3 = 12 s`. The MAE-optimal constant moved
**5.0 → 5.2 s** (lat, MAE 2.3046 → **2.3309 s**) and **3.4 → 4.3 s** (lon, MAE 1.9910 → **2.2696 s**);
`ttm_MAE_skill_s` is measured against **these**, never 0.

### 2.4 ⭐ Power — the ceiling answer, now MEASURED on the real 600-episode val

| axis | S3's `ESTIMATED` projection | **`MEASURED` on the 600** | yield | ≥40 | ≥200 |
|---|---:|---:|---:|:--:|:--:|
| **lateral** | 486 | ⭐ **558 / 600** | 0.81 → **0.93** | ✅ | ✅ |
| **longitudinal** | 438 | ⭐ **520 / 600** | 0.73 → **0.8667** | ✅ | ✅ |

**S3's projection was conservative; the real corpus is better.** Train-side: **2,206 lat / 2,056 lon**
clusters of 2,376.

⚠️ **But the episodes that carry an actual EVENT are far fewer than projected** — and this is the number
that should govern a timing claim:

| axis | projected event clusters | **measured** |
|---|---:|---:|
| lateral | 384 | ⛔ **139** |
| longitudinal | 402 | **312** |

> ⛔ **On the lateral axis only 139 of 600 episodes contain a single manoeuvre-onset event.** The 558
> figure counts episodes with an admissible *decision point*, most of which are correctly labelled
> `t_none`. **A two-arm lateral comparison at parity rests on 139 event-carrying clusters, not 558.**
> That still clears 200 on the *decision-point* definition the spec's bar uses, but **an arm that
> separates only on `t_none` recall has not demonstrated timing skill.** Report both counts or neither.

### 2.5 ⭐ The firewall — S3 **and** S3-W, both reported

`MEASURED`, `s3_blind_baseline_parity_primary.json`. Estimator on every interval: **episode-cluster
bootstrap, `taniteval/ci.py`, B = 2000**, unit = val episode; deltas **paired episode-cluster
bootstrap**. `overlapping_holdout_se` is not used anywhere (verified: `s3_labels.py` imports
`taniteval.ci` directly with no fallback path).

| arm | features | **lat PARITY** [95 % CI] | **lon PARITY** [95 % CI] | lat non-parity |
|---|---|---|---|---|
| **B0** majority | — | **0.0000** (exact) | **0.0000** (exact) | 0.0000 |
| **B1** sensor-only ⭐ = **S3-W bar** | `v0`, in-window v/Δv/κ | **+0.2566** [+0.2075, +0.3090] | **+0.2881** [+0.2280, +0.3475] | +0.1128 |
| **B2** + route | B1 + `route`, `route_graded` | **+0.6292** [+0.5619, +0.6906] | +0.3592 [+0.2951, +0.4204] | +0.3381 |
| ⭐ **B3** FULL conditioning | B2 + `vt_band`, `vt_speed` | **+0.6534** [+0.5896, +0.7119] | **+0.5323** [+0.4780, +0.5832] | +0.3898 |
| **B4** + clock (`H_obs`) | B3 + observable horizon | +0.6561 [+0.5905, +0.7160] | +0.4964 [+0.4411, +0.5472] | +0.3884 |

**Paired deltas** (`paired_episode_cluster_bootstrap`, B = 2000):

| delta | **lat PARITY** | **lon PARITY** | lon non-parity |
|---|---|---|---|
| **B2 − B1** (route echo) | **+0.3727 [+0.3163, +0.4292]** ✅ sep | **+0.0711 [+0.0308, +0.1127]** ✅ **sep** | +0.1657, **not** sep |
| **B3 − B1** (all future conditioning) | **+0.3968 [+0.3426, +0.4492]** ✅ sep | **+0.2442 [+0.1966, +0.2951]** ✅ **sep** | +0.0524, **not** sep |
| **B4 − B3** (clock artifact) | **+0.0027 [−0.0162, +0.0216]** not sep ✅ | ⚠️ **−0.0359 [−0.0634, −0.0108]** **sep, NEGATIVE** | +0.0209, not sep |

### 2.5.1 ⭐⭐ THE SKILL BARS — both variants, on the parity corpus

> | variant | **lateral bar** | **longitudinal bar** | (non-parity, superseded) |
> |---|---:|---:|---|
> | **S3 as specified** (`route` + `vt_*` given) | ⭐ **0.6534** | ⭐ **0.5323** | *0.3898 / 0.2334* |
> | **S3-W** (conditioning withheld; pixels + `v0` only) | ⭐ **0.2566** | ⭐ **0.2881** | *0.1128 / 0.0676* |
>
> **`skill = QWK(model) − bar`, never `QWK(model)`.**
> ⛔ **A lateral QWK of 0.60 is now WORSE than a head with no camera at all.** The bar the brief carried
> (0.3898) is **0.2636 too low** for the parity corpus — an arm scoring 0.55 would have looked like a
> +0.16 win against the old bar and is a **−0.10 loss** against the real one.

### 2.5.2 Refusal conditions, re-evaluated at parity

| rule | verdict at parity |
|---|---|
| **R1 circular** (blind ≥ 0.98) | ✅ **NOT REFUSED** on either axis — 0.6534 / 0.5323 vs a 0.98 threshold. **S3 remains admissible.** Note the margin has halved. |
| ⚠️ **R2 echo** | ⛔ **ARMED ON BOTH AXES NOW** (it was lateral-only). `route`/`route_graded` alone move the lateral blind head **+0.3727 [+0.3163, +0.4292]**, and the full conditioning set moves the *longitudinal* head **+0.2442 [+0.1966, +0.2951]** — both CI-separated. **S3-W is not optional; it is the variant that measures anything.** |
| ✅ **R3 clock** — lateral | ✅ **CLEARED.** +0.0027 [−0.0162, +0.0216], not separated. M1 works. |
| ⚠️ **R3 clock** — longitudinal | ⚠️ **AMBIGUOUS — flagged, not adjudicated.** The delta is **CI-separated (−0.0359 [−0.0634, −0.0108])**, so §7 R3 as *literally written* ("`QWK(B4) − QWK(B3)` is CI-separated **and material**") fires. But it is **NEGATIVE** — adding the clock channel makes the blind head **worse** — whereas §5.3 states the hazard as *"if B4 ≫ B3, the target is partly how much clip is left"*. **The direction R3 exists to catch is absent**, and −0.036 is not material. **The pre-registration does not specify a sign, and I am not choosing one after seeing the result. PI adjudication required.** |
| **R4 / R5 power** | ✅ **BOTH PASS on the 600** (558 / 520 ≥ 200). ⛔ **BOTH FAIL on the published 40** (37 / 34 — see §1.4). |

### 2.6 Sensitivity arm — `H_S3 = 8 s`, at parity (pre-registered)

`MEASURED`, `s3_*_parity_sens_h8.json`:

| | `12 s` (primary) | `8 s` |
|---|---:|---:|
| admissible (lat) | 34,337 | 53,658 |
| clusters (lat) | 558 | 587 |
| **majority rate (lat)** | 0.8553 | ⛔ **0.9116** |
| `t_10_H` occupancy (lat) | 0.81 % | **0.00 %** (empty by construction) |
| admissible (lon) | 24,987 | 38,446 |
| **majority rate (lon)** | 0.6146 | 0.7318 |

**Same read as the non-parity run, more extreme:** a shorter horizon buys coverage and clusters but
**degrades the option set** — at 8 s the lateral problem is **91 % one class** and is a **4-class**
problem whose QWK is not comparable to the primary's. **`H_S3 = 12 s` remains primary**, unchanged.

**Firewall at 8 s** (`s3_blind_baseline_parity_sens_h8.json`), which also settles §2.5.2's R3 question:

| | lat | lon |
|---|---|---|
| B1 (**S3-W bar**) | +0.1949 | +0.2894 |
| B3 (**S3 bar**) | +0.5412 | +0.4789 |
| **clock B4 − B3** | **−0.0075 [−0.0233, +0.0085]** ✅ not sep | ✅ **−0.0011 [−0.0232, +0.0196] NOT separated** |
| R1 REFUSED | ✅ No | ✅ No |

> ⭐ **The longitudinal clock check CLEARS at `H_S3 = 8 s`.** At 12 s it was separated-but-negative
> (−0.0359); at 8 s it is −0.0011 and not separated. **A genuine "how much clip is left" artifact would
> get WORSE at a shorter horizon, not vanish.** This is `MEASURED` evidence that the 12 s longitudinal
> result is a small-magnitude MLP-capacity effect rather than a clock dependence — **but it is
> corroboration, not a licence to overwrite a pre-registered rule.** §2.5.2's flag stands.

### 2.7 What did NOT change — worth stating

* ✅ **R1 does not fire.** The target is not circular on either corpus.
* ✅ **The clock artifact is dead on the lateral axis** on both corpora (M1 works).
* ✅ **The instrument itself is intact:** 20/20 tests pass on pod3, and the mined output from the
  poses-only view is bit-identical to the full cache.
* ✅ **Both power bars still clear on a 600-episode val**, which was S3's operative conclusion.
* ✅ **`ttm` means are stable**: lat 5.204 → 5.406 s, lon 4.066 → 4.712 s. The *distribution shape* of
  the events is similar; what changed is **how many windows have an event at all**.

---

## 3. ⚠️ Escalations — these must not live only in this file

1. ⛔ **The ladder's power bar needs an owner and a per-HP feasibility check BEFORE compute is booked.**
   The 600-episode build is the **hard maximum at parity** (§1.4.1), so **a two-arm HP-x comparison is
   feasible iff its episode yield ≥ 1/3**. This is a one-line arithmetic check per problem and it is not
   currently in the gate protocol. **Owner needed: the dominance-program spec (§0.4 power bars).**
2. ⛔ **The 600-episode val exists on pod2 only, and pod2 is the pod nobody may load.** Until it is
   copied to an eval-capable pod (~30–60 min via the HF relay, after pod2's run finishes), **every
   n ≥ 200 two-arm comparison in the ladder is blocked on a file move, not on science.** The eval pod
   cannot receive it at `/root/valdata` (2.9 GB free). **Owner needed: pod ops.**
3. ⚠️ **`route`, `route_graded`, `vt_band`, `vt_speed` are future-derived and fed at inference to every
   flagship-v4 eval in the program.** S3 measured the consequence for its own target; §2.6 re-measures
   it at parity. **Whether they inflate other reported capabilities is still not established**, and the
   same firewall should be pointed at S1/S2 and at the published route/vtarget metrics.
4. ⚠️ **`physicalai-val-heldout-79d4e3d2d4c6` shares 3 episodes with the published 40** (§1.3). Small,
   and only matters to whoever quotes it as independent. **Owner: the `v4run` / E1c workstream.**
5. ⚠️ **Thread oversubscription silently costs 9× on CPU labelling jobs** (§2.0.1). Not S3-specific.
   Worth a line in the pod runbook rather than being rediscovered.
6. **Firewall duplication is still open.** `s3_blind_baseline.py` is corpus-agnostic and
   `…/2026-07-26-4brain-preconditions/` now exists. **Whichever lands second must delete its copy** —
   this is S3's own escalation, repeated here because it is still true.

---

## 4. Deliverable manifest

**Nothing was `git add`-ed, committed or pushed** (per brief). All paths are in the repo working tree at
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-s3-decision-grade/`.

| Artifact | What | Staged? |
|---|---|:--:|
| `VAL_CEILING_AND_S3_DECISION_GRADE.md` | this report | ❌ not staged |
| `val_corpus_inventory.json` | ⭐ the per-pod / per-root val inventory, counts + byte-level disjointness | ❌ |
| `disjointness_result.json` | the byte-level overlap matrix vs the parity train | ❌ |
| `s3_parity_vs_nonparity.json` | ⭐ every S3 number, non-parity → parity, side by side | ❌ |
| `s3_coverage_parity_primary.json` | coverage funnel + class balance, parity corpus | ❌ |
| `s3_option_set_parity_primary.json` | option set, `ttm` distribution, MAE constants | ❌ |
| `s3_power_parity_primary.json` | power + strata on the real 600-episode val | ❌ |
| `s3_blind_baseline_parity_primary.json` | ⭐ the firewall: B0–B4, **S3 and S3-W bars**, R1/R2/R3 | ❌ |
| `s3_*_parity_sens_h8.json` | the pre-registered `H_S3 = 8 s` sensitivity arm | ❌ |
| `s3_power_MEASURED_published_subsets.json` | measured clusters on the published 40 / pod1 12 / 44 | ❌ |
| `extract_poses_view.py` · `disjointness.py` | the two tools written for this work | ❌ |
| `artifacts/manifest_*.json` | per-episode `sha256(poses)` for every cache probed — the raw evidence | ❌ |
| `artifacts/labels_val_v4_provenance_POD2.json` | pod2's own record: `n_episodes: 600`, `skip_hash f09e44db` | ❌ |

**On the pods (not in the repo):** `pod3:/workspace/s3parity/` — the poses-only views
(`views/physicalai-{train-e438721ae894,val-0c5f7dac3b11}`, 2,376 + 600 episodes), the extraction
manifests, and `out/` with the result JSONs. `pod3:/workspace/TanitAD/s3run/` — the five S3 files
(md5-identical to the repo). `pod2:/tmp/s3val/` — the val extraction scratch, **safe to delete**
(`/tmp`, not MooseFS; ~5 MB).

**Compute honoured:** pod3 only (A40, **GPU never used** — this is a CPU job). pod1, pod2 and the eval
pod were touched **read-only** except for pod2's ~5 MB `/tmp` extraction, which was mmap-based (17 s,
no GPU, no MooseFS write) and was the only way to obtain the 600-episode val the brief asked about.
**No training launched. No commits. No pushes.**
