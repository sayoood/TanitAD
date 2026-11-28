# SR-TTC, Amendment 2 — DRAFT, NOT REGISTERED: a nested, pre-declared LOG subsample of the FIT set, grown only by a fit-set learning-curve rule

**Status: DRAFT (pre-registration design agent, 2026-10-18), for the PI's choice between Amendment 1 item 4 (a) and (b).**
- It becomes binding only if the Master Mind registers it: rename to `PREREG_SELRANK_ARM.amendment2.md` and write its sha256.
- Base texts, both sha256-verified at tip `3a28bc19`:
  - registered body `PREREG_SELRANK_ARM.md`: `cd73c4d4f757dde1953e183be2c0c8ac584d1eb10da8d80515f8356b3d92f0bd`
  - `PREREG_SELRANK_ARM.amendment1.md`: `219c6359604b72d9a403c77de88ff1cdfb3246ced061ac9f4772fb3ac172c0fc`
- **Written before any SR-TTC fit feature, label or score was read.** A2.0 lists every file used.
- **Scope.** This draft settles ONLY Amendment 1 item 4: how much of the fit set S0 must forward.
  - It does **NOT** settle Amendment 1 item 3: dev-gate coverage is 603 of 3,137 sets today, so the gate needs either the full set or a disclosed re-scoping.
  - Item 3 must be added to Amendment 2 before registration. ⛔ Registering this draft alone leaves the dev gate undefined.
- Nothing in the registered body changes: no bar, gate, control, grid, R1 criterion, outcome row or estimator. This draft adds four things:
  - a selection rule
  - a fold rule
  - a fit-set-only scale-up rule
  - the reading of a FAIL under a subsample

## A2.0 What was read, and what was not

| read | what was used | evidence class |
|---|---|---|
| `PREREG_SELRANK_ARM.md`, `.amendment1.md` | full text | — |
| `RESULT.md` (this package) | §3 table, §7 seed floor, §8: **navtest / dev numbers only** | MEASURED there |
| `../2026-10-18-p2-heldout-table/disjointness_proof.json` | counts only: fit = 174,912 sets / 99,495 tokens / **1,167 logs** | MEASURED there |
| `../rl_pool/navtrain_token_table.tsv.gz` (sha256 `c7456c88…cd1`, which matches the proof) | **5 columns only**: `token, log, in_navtest, heldout_log, served_train_sets` (an integer count of sets served per token) | metadata |
| `../2026-10-18-p2-heldout-table/state/unit_*.json` (`n`), `logs/dump_*.log` (`"seconds"`), `MANIFEST.sha256.json` (`bytes`) | timing and file size of the held-out P2 forward | MEASURED there |
| dev-box `data/nuplan/nuplan-v1.1/splits/*.db`, `data/refe_proxy/pixels/*.zip` | **existence and size only**; no file opened | MEASURED (file system) |

**Not read:**
- `served_keys.tsv.gz` (it has `step/rank/ckpt_step` columns beyond keys)
- any label `.jsonl`
- any pre-head feature, logit, PDM value or score of a fit or gate set
- `gate_table.json`, `fidelity.json`, `featcheck.json`
- any dev-gate value

## A2.1 The selection rule (frozen at registration)

Code: `code/srttc_a2_select.py`. Output: `raw/a2_subsample/selection.json` and `raw/a2_subsample/ranked_fit_logs.tsv`.

1. **Unit = LOG**, never token or set, because every estimator in this arm clusters on logs. All sets of a selected log are in; all sets of an unselected log are out.
2. **Fit logs** = logs with at least one token whose `served_train_sets > 0`.
   - **Control, MEASURED:** this reproduces **1,167 logs / 99,495 tokens / 174,912 sets**, exactly Amendment 1. The script aborts otherwise.
3. **Key** = `sha256("SR-TTC/A2/v1" + "\n" + log_name)`. Logs are ordered by ascending key.
   - Hash-based, not RNG-based, so the order is platform-independent and needs no library version.
   - The salt was fixed before the order was computed. It was never re-drawn.
4. **Subsample S_f** = the first `ceil(f × 1,167)` logs of the order, for f ∈ {10 %, 25 %, 50 %, 100 %}.
   - **The subsamples are nested by construction:** S_10 ⊂ S_25 ⊂ S_50 ⊂ S_100.
   - So a scale-up forwards **only the increment**, and no S0 hour is ever wasted.
5. **No stratification.** The only per-log metadata fixed before data are:
   - the log name (date and vehicle);
   - the token and set counts.
   City is not in the name, and its DB is absent on the dev box for 927 of 1,167 logs. Stratifying on counts has no known variance benefit for concordance.
   - Representativeness, reported, not selected on (MEASURED): S_10 holds **8.97 % of tokens and 9.04 % of sets**; S_25 24.0 % / 24.1 %; S_50 53.3 % / 53.5 %.
6. **Parity.** The rule touches only the SR-TTC FIT set, which is a subset of P2's own navtrain on-policy training bank. It does **not** touch:
   - `physicalai-train-e438721ae894` / skip-hash `f09e44db` (a different corpus);
   - navtest;
   - the 24-log held-out GATE set;
   - any other arm's split.
   - **Control, MEASURED:** 0 fit tokens carry `in_navtest` or `heldout_log`.
7. **Missing data (pre-committed).** A selected log whose DB or camera frames exist on **no** host is replaced by the next log in the key order.
   - Every replacement is listed with its reason.
   - If replacements exceed **5 %** of the selected logs, stop and report: the sample would be drifting towards availability.
   - Local availability is **not** a selection input; see A2.8.

Selected-list sha256 (sorted log names, `\n`-joined; MEASURED, `selection.json`):

| f | sha256 |
|---|---|
| 10 % | `9d4d606ad26f871bda37eed5bb3b29b1f91f74526d27433da0105f842265db25` |
| 25 % | `a6c7599ce1d96609447f9a5d2cc3de8ca9188d0dd6023196c1cb78aa37c79d48` |
| 50 % | `325ee346a6a7810aee4ed00e96507097f9ef0682b057b5c78f8fcf4c17f1c64e` |
| 100 % | `4d5c78f1af66d282b7d268b437a653ab5501d8c03aa90fb7b4d1df5a5c594be1` |

**Controls run on the script:**
- A re-run gives byte-identical outputs.
- Changing one `served_train_sets` value makes the sha256 guard fail.
- With that guard bypassed, the count guard fails (174,913 ≠ 174,912).

## A2.2 What each size does to R1's grouped-by-log 5-fold CV

**Fold rule (new, fills a gap in R1):** `fold = rank_in_key_order mod 5`.
- A log keeps its fold as the subsample grows, so CV results at successive sizes are nested too.
- **Minimum k = 20 logs per fold** (design choice). It sits near the 24-log dev gate, the smallest log count this arm already treats as a gate.
- 5 % (59 logs, about 12 per fold) fails k. **10 % is the smallest admissible fraction.**

Counts are MEASURED from metadata (`selection.json`). The last column is ESTIMATED; see the note below.

| f | logs | tokens | sets | logs / fold | sets / fold (min–max) | k ≥ 20 | min logs in a training part | TTC-contrast sets |
|---|---|---|---|---|---|---|---|---|
| 10 % | 117 | 8,922 | 15,808 | 23–24 | 2,505–4,014 | yes | 93 | ≈ 1,325 (≈ 265 per fold) |
| 25 % | 292 | 23,906 | 42,126 | 58–59 | 6,767–10,705 | yes | 233 | ≈ 3,531 |
| 50 % | 584 | 53,052 | 93,636 | 116–117 | 15,668–24,076 | yes | 467 | ≈ 7,848 |
| 100 % | 1,167 | 99,495 | 174,912 | 233–234 | 29,140–42,654 | yes | 933 | ≈ 14,660 |

**The contrast-set column is a HYPOTHESIS.**
- It is `sets × 1,018 / 12,146`, where 1,018 is the navtest tokens on which TTC varies inside P2's top-8 (MEASURED, `RESULT.md` §3).
- That gives 11,443 pairs, or 11.24 pairs per contrast token, so 10 % gives about 14,900 informative pairs.
- Transferring a navtest rate to on-policy navtrain sets is untested.
- Only contrast sets feed the pairwise term. The BCE anchor sees all 64 proposals of every set.

**Resolution at 10 % (ESTIMATED):**
- Navtest's TTC contender concordance is 0.722 [0.686, 0.759] on 1,018 tokens, a half-width of ≈ 0.036.
- The pooled out-of-fold concordance at 10 % (about 1,325 contrast sets over 117 logs) would have a half-width ≈ 0.036 × √(1,018 / 1,325) ≈ **0.032**.
- This assumes equal per-set variance and comparable clustering.
- A paired difference between two heads on the same pairs will be narrower, by an unknown amount (UNVERIFIED). A2.4 is built so that low power costs GPU-hours, never a wrong verdict.

## A2.3 Cost (ESTIMATED; token and set counts MEASURED, rates as stated)

**Rates.**
- **Central, 1.44 s/token** (Amendment 1; the registered §4 rate is 907 s / 629 navtest tokens).
- Corroborated on navtrain: over the 8 held-out units that ran below 2.5 s/token, the marginal OLS slope is **1.411 s/token** (MEASURED, `p2-heldout-table` dump logs). The 2.5 cut is post hoc, and the units it excluded are disclosed next.
- **Contended, 3.045 s/token**: all 11 held-out units together took 1,013.9 s for 333 tokens. Three units ran at 3.9, 11.1 and 23.9 s/token on the contended dev-box GPU. This is a wall-clock bracket, not GPU work.
- **Per-set upper bracket, 1.44 s/set.**
  - Amendment 1's "one pass per token suffices" is INHERITED, not re-verified.
  - Fit sets are on-policy fans from training checkpoints, so if each set needed its own full forward, cost scales with sets.
  - **Pre-S0 check:** time one log both ways.

| f | S0 central (tokens × 1.44 / 3600) | S0 contended (× 3.045) | S0 per-set upper (sets × 1.44) | **increment over previous f** (central) | cumulative (central) |
|---|---|---|---|---|---|
| 10 % | 8,922 × 1.44 / 3600 = **3.57** | 7.55 | 6.32 | 3.57 | 3.57 |
| 25 % | 23,906 × 1.44 / 3600 = **9.56** | 20.22 | 16.85 | 5.99 | 9.56 |
| 50 % | 53,052 × 1.44 / 3600 = **21.22** | 44.87 | 37.45 | 11.66 | 21.22 |
| 100 % | 99,495 × 1.44 / 3600 = **39.80** | 84.15 | 69.96 | 18.58 | 39.80 |

**Not changed by the subsample.**
- S1, the gate forward, is Amendment 1 item 3's question. At the per-token rate, the 1,372 uncovered gate tokens are 1,372 × 1.44 / 3600 = 0.55 GPU-h, and they are blocked on data, not GPU.
- S3, the navtest confirm, stays at **4.86** GPU-h, run only on a dev-gate pass.
- S2, the head fit on cached features, stays negligible. This holds even with 9 configurations × 5 folds × 2 seeds plus the A2.4 learning-curve fits, because the fitted object is one output row on frozen features.

**Cached-feature storage (ESTIMATED):**
- The held-out pre-head file is 10,048,856 bytes for 333 tokens, ≈ 30 KB/token (MEASURED, manifest).
- That gives ≈ 0.27 GB at 10 % and ≈ 3.0 GB at 100 %. Not a blocker.

**Staging: the larger practical cost, and it scales with logs.** MEASURED by file existence on the dev box (`raw/a2_subsample/availability.json`):

| f | logs | DB present | 4 camera zips present | both | tokens in "both" logs | DBs to stage (at the mean 146 MB) |
|---|---|---|---|---|---|---|
| 10 % | 117 | 25 | 38 | **22** | 1,097 | 92 ≈ 13.4 GB |
| 25 % | 292 | 62 | 92 | 47 | 3,501 | 230 ≈ 33.6 GB |
| 50 % | 584 | 120 | 187 | 87 | 7,084 | 464 ≈ 67.8 GB |
| 100 % | 1,167 | 240 | 396 | 187 | 14,807 | 927 ≈ 135.4 GB |

- The mean of 146 MB (median 83 MB) is MEASURED over the 240 local fit-log DBs.
- "4 zips present" is an **upper bound** on frames: the held-out table found zips present but needed frames missing (1,010 tokens). Frame completeness is UNVERIFIED.
- ⇒ **S0 cannot run on the dev box as it stands, at any f.** It needs the host where P2 trained, or staging. Note also that the dev box's D: is exFAT with a large allocation unit.

## A2.4 The scale-up rule: read on the FIT set only, BEFORE the gate, so the gate is still read once

Start at **f₀** (the PI's choice; recommended 10 %), with an **envelope f_max** (the PI's cap; recommended 100 %).

At each size f:

1. Run S0 on the increment only. Run R1 as registered: the 9-configuration grid, choose `c*` by mean out-of-fold contender concordance, with folds from A2.2. Write `c*` and its sha256.
2. **Learning curve (LC), at `c*`, seeds 0 and 1.**
   - Refit on nested prefixes of each fold's training logs, taken in key order: p ∈ {25 %, 50 %, 100 %}.
   - Score the pooled out-of-fold contender concordance of the TTC row on the **same** out-of-fold pairs. That is P2's frozen top-8 per set, pairs with differing realised TTC, logit ties counted as 0.5.
   - **Statistic:** Δ = c(100 %) − c(50 %), one doubling of training logs.
   - **Estimator:** paired log-cluster bootstrap over the S_f logs (each is out-of-fold exactly once), **10,000 resamples, seed 20260927** (`zero_tail.Boot`), percentile 95 %.
   - It answers "another draw of logs" only. The two training seeds answer training noise (H-ESTIM-SEED-1). Inference is deterministic.
   - The p = 25 % point is descriptive. ⛔ **No learning-curve exponent is fitted or quoted, and nothing is extrapolated.**
3. **Call, both seeds required:**
   - **RISING:** the lower bound of Δ is above 0 on both seeds.
   - **FLAT:** the upper bound of Δ is below **δ = 0.010** on both seeds.
   - **INCONCLUSIVE:** anything else, including the two seeds disagreeing.
   - Why δ = 0.010 (a design choice, HYPOTHESIS-grade, frozen at registration): it is below a third of the navtest concordance resolution (half-width ≈ 0.036, MEASURED), so a doubling that buys less is not a lever the confirm could see.
4. **Action:**
   - **FLAT** ⇒ freeze `c*`, write its sha256, read the dev gate **once**.
   - **RISING or INCONCLUSIVE** with a next f ≤ f_max ⇒ scale to the next size in {10, 25, 50, 100} % and repeat from 1.
   - At f = 100 %, or at f_max ⇒ read the dev gate once, and record the LC call at that size.
5. **The asymmetry is deliberate.**
   - FLAT needs a *narrow* interval, so an under-powered LC can never declare "more data will not help".
   - Low power routes to more data, which costs GPU-hours and never produces a wrong verdict.

Each step's `c*`, LC intervals and call are banked **before** the next step starts. The 8 unchosen configurations at every size are never scored on the gate or on navtest (R1, unchanged).

## A2.5 What the subsample changes, and what it does NOT

**Does NOT change (each is set by an evaluation set or the seed floor, not by fit size):**
- **The navtest MDE ≈ 0.17** (ESTIMATED, registered §6: 0.43 × √(1,000 / 6,408)).
  - That is the CONFIRM set's resolution.
  - ⚠️ The brief that commissioned this draft called it the "dev-gate MDE". It is not: §6 computes no dev-gate MDE ("24 logs cannot support a lower-bound rule").
- **The dev-gate bar of +0.30 on both seeds.** It comes from the training-seed range 0.33–0.44 on the dev pick (MEASURED, `RESULT.md` §7).
- The dev and navtest sets, both read once. The navtest bars, controls C0–C5, the frozen fan and selection rule, the grid, R1's criterion, the estimator. S1 and S3.

**Changes:**
- **Learnability.** Less fit data can lower `f` in the reach table (§6). `f` had no measured anchor, so nothing registered moves, but a subsample-fit FAIL is not a full-data FAIL until A2.4 says the data was not binding.
- **Training-seed variance of the fitted row:** likely larger on less data (HYPOTHESIS).
  - C3, the two-seed requirement, carries it.
  - |seed 0 − seed 1| on the dev gain is **reported** against the 0.44 floor. This is a disclosure, not a new bar.
- **Fold sizes** (A2.2), and the **C2 shuffled control** runs at the final fit size.
- **Representativeness.** 117 of 1,167 logs at 10 %. The city mix is unchecked (UNVERIFIED; it needs the DBs).

## A2.6 How a FAIL is read: committed before any data (extends registered §7; Rule Zero)

| LC call at the gate size | dev gate | reading | next lever, executed in the same run where it needs no new GPU |
|---|---|---|---|
| any | PASS, both seeds | as registered ⇒ navtest CONFIRM, read once | registered §7 rows. If the LC call was not FLAT, add **(iv) refit at the next nested size** to ADOPT-SIGNAL and NOT PROVEN: a larger `f` is the cheapest way to grow a NOT-PROVEN effect past the ≈ 0.2 resolution |
| **FLAT** | DEV-FAIL | **HEAD/FEATURE-LIMITED.** More fit data of this kind is **not** the lever (MEASURED plateau). The subsample is **not** the explanation, and it must not be offered as one | registered §7 feature-vs-capacity probe: CPU, minutes, on the cached S_f features with the A2.2 folds. If its contender concordance stays ≤ 0.72, the lever moves to the trunk (P6 pattern) |
| **RISING / INCONCLUSIVE** (only reachable when f_max < 100 % stopped the scale-up) | DEV-FAIL | **DATA-LIMITED. NO verdict on H-SELRANK-TTC.** Logged as *not a refutation* | (1) the feature-vs-capacity probe, run anyway because it is CPU only. (2) The next nested increment: GPU-h from the A2.3 increment column, blocked on the PI's cap, **named**. After it, the dev gate is re-read at the larger size. That is **gate read #2, disclosed, the maximum**. The navtest CONFIRM is still read once |
| any | — | REFUTED (navtest upper bound < 0) | unchanged from §7 |

⚠️ **Note for the registrar (flagged, not changed here).**
- §7's 0.72 threshold is P2's **navtest** top-8 TTC concordance.
- On fit-set CV, the like-for-like reference would be P2's frozen TTC row on the same out-of-fold pairs. But that is **in-sample for P2**, which trained on these sets, so it is biased upward.
- The two references answer different questions. Whichever is used, it should be named at registration.

## A2.7 Both outcomes for the PI's choice itself, committed

| PI chooses | worst-case S0 (central) | best-case S0 | dev-gate reads | a FAIL is interpretable? |
|---|---|---|---|---|
| (a) full set up front | 39.80 | 39.80 | 1 | yes, but there is no LC to separate head-limited from data-limited except at one size; the A2.4 LC can still be run at 100 % |
| **(b) f₀ = 10 %, f_max = 100 %** (recommended) | **39.80** (= (a): nested, no waste) | **3.57** | **1 in every branch** | yes: FLAT ⇒ head-limited, otherwise the data is scaled before the gate |
| (b′) f₀ = 10 %, f_max < 100 % | the cumulative cost at f_max (A2.3) | 3.57 | 1, or 2 in the DATA-LIMITED branch (disclosed) | yes, with the DATA-LIMITED row reachable |

The only things (b) costs over (a) are **calendar time** (up to four S0 rounds) and up to four **staging rounds**.

## A2.8 Considered and set aside

| alternative | why not |
|---|---|
| **Logs already on the dev box** (187 logs / 14,807 tokens with DB plus four zips; ≈ 5.9 GPU-h, zero staging) | What is on the box was set by earlier packages' needs, not by a pre-declared rule. It is not content-independent (UNVERIFIED either way). It fails the "deterministic rule fixed before data" requirement. Admissible only if its provenance is shown content-independent |
| Token or set subsample | breaks log clustering; the estimator and R1 group on logs |
| An RNG draw with a seed | the order would depend on the library and its version; a salted hash does not |
| Stratify on city | not in the log name; the DB is absent on the dev box for 927 of 1,167 logs |
| Reading the dev gate after each scale-up | multiplies gate reads; A2.4 decides scale-up on the fit set only |

## A2.9 Deliverables of this draft

- `code/srttc_a2_select.py`: the rule, with three guards (table sha256, counts reproduce Amendment 1, 0 navtest or held-out flags).
- `code/srttc_a2_availability.py`: existence and size only.
- `raw/a2_subsample/selection.json`, `raw/a2_subsample/ranked_fit_logs.tsv` (rank, log, key, fold, first fraction, token and set counts), `raw/a2_subsample/availability.json`.
- `raw/a2_subsample/NOTES.md`: the PI question.
- This file and its `.sha256`.
