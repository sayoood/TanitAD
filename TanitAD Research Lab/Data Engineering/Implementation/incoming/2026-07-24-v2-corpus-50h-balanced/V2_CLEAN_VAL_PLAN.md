# C64 option B — a clean validation split for the v2 line

**Written 2026-07-29 05:1x UTC, while v2corpus is at step 26,200 / 30,000.** This is the second of
the two paths the PI must choose between (option A = score on the 19 leak-free episodes, banked as
`v2bal_leakfree_val19.json`). Prepared in advance so the choice is between two ready things.

## Why option B may be the right answer

C64 found that **21 of 40** canonical validation episodes sit inside v2corpus's training corpus. The
19 survivors clear the harness bar, but each of the three caveats weakens the contrast:
19 clusters widen the paired bootstrap; the survivors are what a manoeuvre-balanced selection left
behind and may skew toward lane-keeping *against* the v2 arm; and v1 must be re-scored there anyway.

⭐ **The deeper point: `physicalai-v2bal` was always intended as the corpus for the NEXT flagship
generation.** A generation needs a validation split disjoint from *its own* training selection.
Borrowing v1's split was never going to be right for more than one comparison.

## ⭐ The clean split costs NO new data

**MEASURED 2026-07-29 from `v2_pool_scored.parquet` and `r0_selection_v2.parquet`:**

| quantity | value |
|---|---|
| scored pool | **18,987 clips** |
| v2bal selection (train) | **9,000** |
| ⭐ **UNSELECTED REMAINDER** | **9,987 clips** |

**9,987 clips the v2 selection never touched.** A v2 validation split drawn from that remainder is
**disjoint from v2corpus's training set by construction** — no new download, no new ingest, and the
disjointness is provable from two parquet files rather than argued.

⚠️ **CORRECTION carried here:** the pool has been quoted repeatedly (including in `C64` and my chat
reports) as **18,731**, taken from `V2_CORPUS_DESIGN.md`'s *"18,731 moving clips"*. The scored pool
parquet has **18,987** rows. The selection fraction is therefore **9,000 / 18,987 = 47.4 %**, not
48.0 %. **The C64 base-rate argument is unaffected** — 47.4 % still agrees with the 52.5 % observed
overlap, which is the point that argument turns on — but the figure should be right wherever it is
repeated.

## ⭐ FEASIBILITY — MEASURED 2026-07-29 08:1x UTC (`v2_clean_val_feasibility.json`)

**The concern this plan raised qualitatively is now QUANTIFIED — and it is large.**

| feature | v2 **TRAIN** | **REMAINDER** |
|---|---|---|
| `junction` | **0.6134** | **0.2494** |
| `mean_v` | 10.18 | **13.56** |
| `stop_frac` | 0.0823 | **0.0376** |
| `cum_head` | 78.33 | **31.92** |

⇒ **The remainder is the RESIDUE of a manoeuvre-balanced selection**: 2.5× less junction-heavy,
faster, less stopping, 2.5× less heading change. **A UNIFORM draw from it would produce a val set
nothing like the train distribution** — which is precisely why step 2 below says *stratify*.

✅ **VERDICT: FEASIBLE.** `junction` is the binding stratum — a 600-clip val needs **368** junction
clips and the remainder holds **2,491**: **6.77× headroom.**

⚠️ **THE COST, and it must be reported per-stratum, never as a total N.** Matching **oversamples the
scarce strata**, so the effective *independent* sample shrinks on exactly the manoeuvres that matter
most. A 600-clip val whose junction rows are drawn from a 6.77×-thin pool is **not** 600 independent
junction observations.

⛔ **UNRESOLVED — do NOT feed these to a selector yet.** `stopped`, `city` and `hw` are not clean
0/1 columns in `v2_pool_scored.parquet`, and **`lk` is not a rate at all** (train mean **80.55**,
remainder **120.24** — a count-like column). My first pass computed `needed_in_val = 48330` for it,
which is **nonsense arithmetic on a misread column**. **Each column needs its semantics established
before it becomes a stratification axis.**

## Design of the split

1. **Draw from the remainder only.** Candidate set = `pool.clip_id − selection.clip_id` (9,987).
2. **Match the v2 TRAIN distribution, not the pool's.** The v2 corpus is deliberately
   manoeuvre-balanced (turns 28.0 %, lane_keep 45.0 %, junction presence 61.3 %). A val split drawn
   *uniformly* from the remainder would be **richer in lane-keeping than the train set**, which is
   exactly the bias that makes the 19-episode surface unattractive. Stratify on the same axes the
   selector used (`junction`, `net_head`/`cum_head`, `mean_v`, `stop_frac`) so val mirrors train.
3. **Size: 600 clips**, matching the parity line's val (`physicalai-val-0c5f7dac3b11`, 600 eps) so
   interval widths are comparable to the numbers the programme already quotes.
4. **Emit the exclusion proof as a build artifact** — this is C64's own rule:
   `val_clip_ids ∩ train_clip_ids` must be printed and committed, **even when zero**.
5. **Freeze and hash it** the way the parity corpus is frozen: a committed manifest with a clip
   sha256, so the trainer's preflight can verify it and refuse on mismatch.

## ⛔ What this does NOT give us

- ⛔ **It does not make v2corpus comparable to v1.** Different corpora *and* different val surfaces.
  A v2-line number on a v2-line val is an **internal** baseline for that generation — it cannot be
  placed beside 0.4271 in `MODEL_REGISTRY.md`.
- ⇒ **If the goal is specifically "did the 50 h balanced corpus beat the 13 h parity corpus?", option
  B does not answer it.** Only option A does, on 19 episodes, with its caveats. **The two options
  answer different questions, and that is the real choice** — not which is more rigorous.
- ⚠️ A new val split cannot be validated against any existing published number, so its first use is
  also its calibration. Expect the first v2-line result to be uninterpretable in isolation.

## Recommendation

**Do both, in this order** — they are not exclusive and the first is nearly free:

1. **Option A now** (~0.3 GPU-day): score both arms on the 19, report `leak_free_n = 19` in the
   headline with all three caveats. It is the only path that speaks to the corpus question, and the
   episode list is already banked.
2. **Option B before any further v2-line training** (~1 engineer-day, 0 GPU for the selection):
   the next generation needs its own split regardless of how A turns out, and building it after more
   v2 arms exist would mean re-scoring all of them.

⚠️ **Neither is authorised as a corpus/HF change without the PI.** This is a plan, not an action.

## Evidence class

| claim | class |
|---|---|
| pool 18,987 · selection 9,000 · remainder 9,987 | **MEASURED (ours)** — read from the two parquets this session |
| the 18,731 → 18,987 correction | **MEASURED** — supersedes the figure quoted in C64 and in chat |
| "a remainder-drawn split is disjoint by construction" | **MEASURED** — it is a set difference, provable from the two files |
| the stratification design | **PROPOSED** — not implemented, not validated |
| "the first v2-line result will be uninterpretable in isolation" | **HYPOTHESIS** |
