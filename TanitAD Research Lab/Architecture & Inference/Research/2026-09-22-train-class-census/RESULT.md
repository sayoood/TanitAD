
<!-- TRAIN-CLASS-CENSUS-MEASURED-H-BOXCLS-1-UNBLOCKED-2026-09-22 -->

### ⭐⭐ 2026-09-22 — the TRAIN class census is MEASURED and `H-BOXCLS-1`'s weight vector exists: the join I called missing was on this box all along

MEASURED by me, CPU only, read-only
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-train-class-census/`).
See `RETR-2026-09-22-TRAIN-JOIN-EXISTS` for the blocker I reported that was never there.

**Source:** `C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz` (136.7 MB,
2026-09-05) — the canonical train join whose summary `CLAUDE.md` already cites.

⭐ **THE CONTROL FIRST, because a partial read would have produced a plausible weight vector from a
partial corpus** — and the whole reason an eval proxy was refused is that class frequencies are
sampling-sensitive. The census must reproduce the join's own summary **exactly**:

| | meta says | I counted |
|---|---|---|
| episodes / frames / boxes | 2,308 / 433,040 / 12,122,129 | **2,308 / 433,040 / 12,122,129** ✅ |

**THE TRAIN CENSUS** (12,112,052 boxes in vocabulary; 0 boxes without a `cls`):

| class | count | share | weight (inv-freq, mean 1) |
|---|---|---|---|
| automobile | 9,274,706 | 76.574 % | **0.004084** |
| person | 1,926,364 | 15.904 % | 0.019664 |
| heavy_truck | 357,048 | 2.948 % | 0.106093 |
| rider | 190,793 | 1.575 % | 0.198541 |
| trailer | 164,688 | 1.360 % | 0.230012 |
| bus | 89,781 | 0.741 % | 0.421918 |
| protruding_object | 51,124 | 0.422 % | 0.740948 |
| other_vehicle | 35,543 | 0.294 % | 1.065757 |
| stroller | 13,346 | 0.110 % | 2.838319 |
| animal | 8,659 | 0.072 % | **4.374664** |

⇒ **imbalance 1,071.1 : 1** (`automobile` : `animal`). ⚠️ That is **NOT** the 577.5 : 1 I have been
quoting all session — that figure was the **eval** matched-pair sample, and the real train imbalance
is **1.86× worse**. Every statement of the collapse's severity that used 577.5 was understated.

⭐ **AND IT SETTLES THE PROXY QUESTION WITH DIRECT EVIDENCE.** `e172c65` refused a held-out eval
proxy on the strength of eval's *internal* variance (2.26× across two 62-clip halves). With the
train vector now measured, the eval halves differ from it by up to **2.167×** (`rider`). **The
refusal was right**, and it is now backed by the comparison itself rather than by an inference.

⛔⛔ **A VOCABULARY FINDING, AND `CLAUDE.md` IS WRONG ABOUT IT.** The train join carries
**`train_or_tram_car`: 10,077 boxes** — a class **absent from `AGENT_CLASSES`**. `CLAUDE.md`'s
`refc_agents` warning states that a draft vocabulary containing `bicycle`, `motorcycle` and
`train_or_tram_car` named classes *"**none of which exist in the corpus**"*. **`train_or_tram_car`
does exist**, 10,077 times, in the corpus we train on.
⭐ **The CODE is safe and was verified, not assumed:** `targets_from_join(classes=[…])` maps
`train_or_tram_car` → **−1**, and `slot_set_loss` masks `ok = ct >= 0`, so those boxes are
**excluded from the class term rather than relabelled**. MEASURED by round-tripping the three
strings: `automobile → 0`, `train_or_tram_car → −1`, `person → 5`. ⇒ the defect is in the **steering
prose**, not the model; 0.083 % of train boxes are silently unsupervised for class, which is the
correct handling but should be a stated one.

⇒ **`H-BOXCLS-1` loses one of its two blockers.** The weight vector is measured, reproducible and
banked. What remains is the PI's GPU call.
