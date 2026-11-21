# SFT-1 result: scorer-only fine-tune on the teacher's labels: NOT PASSED (final read missing, run OOM-killed)

> Pre-registration: `eval/PREREG_SFT1.md`. Raw: `raw/2026-10-04-sft1/sft.jsonl` (md5 `8894db42…`, the pod's full event
> log), `raw/2026-10-04-sft1/pod_chain_and_gate.txt` (chain log, PREFLIGHT_PASS token, last lines).
> Tier: stage 1 is a label-only held-out read (no closed loop). Interval: paired log-cluster bootstrap over the 24
> held-out logs, 10,000 resamples, seed 20260927; it answers *another draw of logs* only (one fine-tune seed).

## Verdict

**Neither arm passes stage 1; neither goes to navtest.** The registered rule is *lower bound > 0 at the final update*.

| arm | last read (update 2,400 of 2,733) | vs base, ×100 [95 % CI] | picks = 0 (base 169) | AUC NC / DAC (base 0.884 / 0.745) |
|---|---|---|---|---|
| A: per-component BCE (control) | 84.84 | **+0.01 [−0.20, +0.25]** | 175 | 0.880 / 0.740 |
| B: BCE + 0.3 ListNet | 84.19 | **−0.65 [−1.47, +0.27]** | 179 | 0.867 / 0.734 |

MEASURED, `raw/2026-10-04-sft1/sft.jsonl`. The base pick is 84.84; random 73.12; best of 64 97.60.

## ⛔ Deviation: the final-update read does not exist

The run was **OOM-killed at update 2,550 of 2,733** (`ZZEXIT 137`, 14:50:43 UTC). **I caused it.** The SFT-4 launch-gate
smoke loaded the full 177,836-set bank, plus 3 DataLoader workers, beside the live fine-tune and 28 CPU relabel
workers. That pushed the pod's 50 GB cgroup over its limit (`memory.events` oom_kill 2 → 4). The fine-tune was the largest
process, so the kernel killed it. The smoke died too. The final read and the final checkpoints (`save_full`) were never
written.

The 2,400 read is the last one, and the stage-1 verdict stands on it. It is decision-equivalent rather than formally the
registered read:
- **ESTIMATED from the logged schedule.** The updates after 2,400 carry 0.31 % of the run's learning-rate mass; those
  after the kill carry 0.05 %. The schedule is verified: the computed lr(2525) equals the logged 4.4269e-07 exactly.
- **MEASURED.** The 2,000 → 2,400 window carried 2.87 %, about 9× more, and moved A by +0.07.
- For A to pass, its mean would have to rise by about +0.21 at an unchanged interval width.
- B has had a negative mean at every one of its six reads.

`scorer_states_last.pt` (update 2,400, md5 `f9a2fd78…`) survives on the pod. A full checkpoint can be rebuilt from it
offline if one is ever needed.

## What the trajectory says (MEASURED, 7 reads)

- **A never separates from the base.** Its mean ranges from −0.28 to +0.02 across all reads. Its within-set AUCs stay
  at or slightly below the base's.
- **The ranking loss hurts.** B runs −0.55 to −0.92, and both its AUCs fall.

Ranking harder against the teacher's labels makes the selection worse on a truth built from the same family of labels
(held-out v5: teacher calculators with the NAVSIM drivable area). This supports `REVIEW_7_GAP_TO_PAPER.md`'s diagnosis:
the labels limit selection, not the optimisation.

## Rule Zero: the next arms (already built and running)

- **SFT-3** (`eval/PREREG_SFT3.md`): the expected-score objective on the teacher labels. It started automatically at
  14:52:18 UTC when this run exited.
- **SFT-4** (`eval/PREREG_SFT4.md`): the paper's own labels (NAVSIM PDM targets of the executed plans) plus the teacher
  lane output. Its chain is armed behind SFT-3 and the PDM training relabel.

## The incident's class (logged in RETRACTION_LOG)

**Two jobs sized against the wrong scope.**
- **Memory:** a second bank-loading process was launched beside a live run on a 50 GB cgroup without pricing its
  footprint. Each fine-tune holds the bank in the main process and 3 forked DataLoader workers.
- **CPU:** the pod reports `nproc` 96, but its cgroup quota is **7.65 CPUs** (`cpu.max 765000 100000`;
  `nr_throttled` 1.28 M of 7.39 M periods). 28 relabel workers were sized against the host count.

**Fixes applied the same hour:**
- every relabel process now has `oom_score_adj` 1000, so the kernel kills relabel workers before any fine-tune
  (the fine-tune reads 669, the workers 1,334);
- their autogroups run at nice 19;
- the SFT-4 smoke now loads a label-carrying subset bank (`refe/smoke_bank_subset.py`) with one DataLoader worker, and
  launches only after a memory-headroom check.
