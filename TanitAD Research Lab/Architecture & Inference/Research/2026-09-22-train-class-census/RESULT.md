
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

<!-- CORR-2026-09-22-VOCAB-ATTRIBUTION -->

### ⛔ CORRECTION (same day, 2026-09-22) — the vocabulary finding above is REAL but MIS-ATTRIBUTED and OVERSTATED AS NEW

The block above says: *"⛔⛔ A VOCABULARY FINDING, AND `CLAUDE.md` IS WRONG ABOUT IT … `CLAUDE.md`'s
`refc_agents` warning states…"*. Three things in that sentence need fixing, and only the
measurement survives untouched.

**1. `CLAUDE.md` does not contain it.** MEASURED: `CLAUDE.md` is 1,269 lines and contains
**zero** occurrences of `train_or_tram_car`, of `none of which exist`, and of `refc_agents`
(read control: the same command reports the file's 1,269 lines, so this is an absence in a file
that WAS read, not a failed read). The sentence lives in **three CODE files**:

| file | form |
|---|---|
| `stack/tanitad/refs/refc_agents.py:22` | module docstring |
| `stack/scripts/refcv5_preflight.py:126` | `check_class_enum` docstring |
| `stack/tests/test_refc_agents.py:32` | test docstring |

⚠️ Root cause: **`A SUMMARY IS NOT A PATH`** — I quoted the source from memory instead of from
the file, and "a warning about `refc_agents`" became "`CLAUDE.md`'s `refc_agents` warning". The
programme has a memory note for this exact failure and I reproduced it four days later.

**2. The EXISTENCE of `train_or_tram_car` was already registered, so "⛔⛔ A FINDING" overstates
it.** `D-CLEARANCE-IS-AGENT-NOT-INFRA-1` records it as the 11th `obstacle.offline` label —
69 boxes on B1 eval, **2,419 measured 2026-07-27**, 275 on a disjoint 193-clip train set — and
even notes that `refc_agents.py` asserts it does not exist. ⇒ **What is genuinely new is the
TRAIN-split magnitude and the handling proof**, not the existence: **10,077 boxes = 0.0831 % of
the canonical TRAIN join**, and the round-trip showing they are masked rather than relabelled.
State it that way.

**3. The claim is false for ONE of the three names, not all three** — and the precise version is
the stronger one. MEASURED over 12,122,129 TRAIN boxes: `bicycle` **0**, `motorcycle` **0**,
`train_or_tram_car` **10,077**. Two thirds of the original sentence were correct.

⭐ **WHAT IS FIXED, AND WHAT DELIBERATELY IS NOT.** The prose at all three sites now states the
measured counts and the masking behaviour. ⛔ **The GUARDS ARE UNCHANGED and remain right**:
`train_or_tram_car` must stay out of `AGENT_CLASSES` because `AGENT_CLASSES` **is**
`bev_raster.ALL_CLASSES` and has exactly one spelling — not because the label cannot occur.
`targets_from_join` maps it to `-1` and `slot_set_loss` masks `ok = ct >= 0`, so those boxes are
excluded from the class term, never relabelled. Pinned by
`test_cls_weight_stamp.py::test_an_out_of_vocabulary_class_is_masked_not_relabelled` and
mutation-proven by arm **S9**, which relabels the unknown class to `0` and must go RED.

⚠️ **The class worth keeping: a guard whose stated JUSTIFICATION is false still passes, so
nobody ever re-reads it.** A reader who believes the label cannot occur never asks what happens
to it — and the answer was a real, quantified property of the training signal (0.0831 % of TRAIN
boxes carry no class supervision). This is the sibling of *"a check that shares the defect it
checks for"*: here the check was correct and its REASON was not, which is harder to see because
nothing ever fails.
