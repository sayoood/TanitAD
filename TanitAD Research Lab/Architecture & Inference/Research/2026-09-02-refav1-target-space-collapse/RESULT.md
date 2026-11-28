# RESULT — the shipped anti-collapse configuration PREVENTS refav1's collapse

`E-ARCH-TSC-1` · 2026-09-02 · Architecture & Inference · Thor · **Tier T0** (mechanism probe,
250 steps/arm — *not* a capability claim, no driving metric produced)

## Verdict

⭐ **CONFIRMED on every pre-committed read.** The collapse is real, reproducible on demand, and
the shipped configuration stops it. refav1 may relaunch for a full epoch.

⚠️ **Wording, per SPEC §4a:** this licenses *"the **shipped anti-collapse configuration**
prevents the collapse"*. It does **NOT** license *"frozen targets prevent the collapse"* — arm B
carries **two** changes (`target_space` **and** `detach_aux_targets`). Attribution between them
needs arm **C** (`frozen` + `--no-detach-aux-targets`) and is not claimed here.

## The reads

| read | A — regression (`adapter`, no detach) | B — fix (`frozen` + detach) | |
|---|---|---|---|
| **R1** `adapter_std` | **0.4763 → 0.3062 (−35.7 %)**, 9/10 deltas down | **0.4778 → 0.5937 (+24.3 %)**, 2/10 down | ✅ |
| **R2** `tgt_std_op` (known value) | — | **0.9995 → 0.9925**, pinned ~1.0 | ✅ admissible |
| **R3** `tgt_std_tac` | 0.0571 → 0.0247 (**−57 %**) | 0.0571 → **0.8418 (+1374 %)** | ✅ |
| **R3** `tgt_std_str` | 0.0987 → 0.0220 (**−78 %**) | 0.0987 → **0.3421 (+247 %)** | ✅ |
| **participation** (floor **8.56**) | 34.09 → **7.43**; below floor at steps **200, 225, 250** | 34.09 → **16.65**; **never below floor** | ✅ |
| **R4** regression reproduces | **yes — decisively** | — | ✅ |

⭐ **The single most telling pair:** in A the loss fell **0.600 → 0.215 while the strategic target
shrank 78 %** — improvement bought entirely by the target evaporating. In B the loss fell
**1.752 → 0.676 while the tactical target GREW 15×**. Same falling curve, opposite meaning. That
is precisely why the target-scale instrument had to exist before either arm ran.

## Stability, not just endpoints

`participation` in B falls 34.09 → 16.65 (−51 %) and then **holds**: second-half mean **14.51**,
last six readings 13.7 · 15.0 · 14.3 · 14.1 · 13.3 · 16.6 — flat. A's second-half mean is **9.58**
and its last six are still descending (15.6 · 10.9 · 9.6 · 7.2 · 6.7 · 7.4), through the floor.
⇒ B settles into a structured representation; A does not settle at all. The initial 34.09 is a
random-init artifact in both arms, so *some* fall is expected — the question was whether it stops,
and in B it does.

## ⚠️ R1's criterion was MIS-SPECIFIED, and I am not force-fitting it

The SPEC committed *"B holds `adapter_std` within **5 %** of its start"* for CONFIRMED and
*">15 % fall"* for REFUTED. **B rose 24.3 %** — neither branch. A rise is unambiguously the
healthy direction (the representation expanded rather than contracted), so the finding is not in
doubt; but the outcome table had **no branch for the observed direction**, which is exactly the
defect logged against `PREREG_MM_E19_K60_HORIZON` hours earlier, repeated by me in a SPEC written
after logging it. ⇒ **the criterion should have read "does not fall"**, and the standing rule
stands: *every outcome table carries a branch for each direction, including the one you do not
expect.*

## What this does NOT settle

* **Attribution** between `frozen` and `detach_aux_targets` — needs arm C.
* **The 30-step untruncated BPTT chain** (`refa_v1.py:525/529`). Unaddressed by either arm. A's
  gnorm excursions on the earlier run (3.6e3 · 5.7e7 · `inf`) are its signature, and the Lab's
  ASK-1 found **no published recipe back-propagates that far**. Separate mechanism, separate fix.
* **SigReg and the variance floor** remain **off** (weight 0). Neither was needed to stop this
  collapse; they stay available and calibrated (⛔ SigReg's raw term reads ~238 on this scale —
  do not copy v6's 0.1).
* **Any capability claim.** 250 steps at T0.

## Deliverable manifest

| artifact | location |
|---|---|
| this result | `…/2026-09-02-refav1-target-space-collapse/RESULT.md` |
| spec + amendment | `…/2026-09-02-refav1-target-space-collapse/SPEC.md` |
| arm A raw | Thor `/home/nvidia/experiments/TSC-A-adapter/train_log.jsonl` → `raw/TSC-A-adapter.jsonl` |
| arm B raw | Thor `/home/nvidia/experiments/TSC-B-frozen/train_log.jsonl` → `raw/TSC-B-frozen.jsonl` |
| mechanisms + default flip | `97e8019` |
| instrument correction | `e1352da` |
