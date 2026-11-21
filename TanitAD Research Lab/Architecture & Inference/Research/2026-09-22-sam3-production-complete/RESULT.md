
<!-- SAM3-CORPUS-PRODUCTION-COMPLETE-2026-09-22 -->

### ⭐⭐ 2026-09-22 — SAM3 CORPUS PRODUCTION IS COMPLETE: 4,719/4,719, manifest `COMPLETE`, and the §4 plan is FINAL at 83 clips

MEASURED by me, read-only
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-sam3-production-complete/`).
The 4,719-clip v7 corpus augmentation the PI asked for on 2026-09-15 has finished.

| | |
|---|---|
| clips | **4,719 / 4,719 (100.0 %)** |
| DONE marker | `2026-09-22T17:39:14+02:00` (Berlin — the frame is IN the file) |
| OK / gate-failed | **4,569** / **150** |
| publisher final run | `2026-09-22T18:21:55 … status **COMPLETE**, commit 312b0a91fa` |
| on HF | 4,569 published + **108** flagged = **4,677 maps** |
| not published | **42**, all `GIVEN_UP FAIL-EXPORT-CHECKS` |
| flagged split | **83** "near path unlabelled…" + **25** "path untestable (parked / stopped ego)" |

Arithmetic closes: 4,569 + 150 = 4,719; 150 = 108 flagged-and-published + 42 given-up.

⛔⛔ **`ZZSTATUS` READS `CHECK`, AND THAT IS THE CORRECT POST-COMPLETION STATE — NOT A FAULT.**
`alive` is `{supervisor: [], driver: [], feeder: null, publish_loop: []}`: every component has
exited, which is exactly what should happen after DONE. ⭐ **The restart condition is CONJUNCTIVE —
*no lock holder AND no DONE file* — and there IS a DONE file.** A naive read of `CHECK` would have
restarted the supervisor on a finished corpus and begun re-producing it. The DONE marker is the
discriminator, and diagnosing before acting is what the runbook demands for exactly this reason.

**THE §4 PLAN IS NOW FINAL** (`s4_retier.py`, DONE detected, `production_complete: true`):

| | |
|---|---|
| flagged entries | 108 |
| **selected for re-tier** | **83** |
| **refused** | **0** — the per-clip premise holds on every one |
| compliant share under (a) | **exactly 1.0** |
| (c) withheld share | min **0.100077** · median **0.137173** · mean **0.175081** · max **0.485261** · **0** above 0.5 |

⚠️ **Projection tracking, reported as measured rather than as a success:** the prereg projected
**78** at completion from `s4_price.json`; the actual is **83** — **+6.4 %**. The projection was
close and **slightly low**. It tracked 54 → 77 → 80 → 83 as the corpus grew.

⛔ **NOTHING APPLIED.** The layout question is PI queue item 19, and the default I had offered was
**withdrawn** in `d247b0e` — a default I invent is not a ruling he made. `--layout` sits at
`pending`, and `A`/`B`/`C` all refuse without his word. ⚠️ Note the publish loop has now **exited**,
so a manifest edit would not reach HF on its own: applying the judgement is a deliberate publish,
which is a second reason it waits.

⇒ **What the corpus is today:** 4,677 maps on HF under a manifest marked COMPLETE, of which 83 are
tiered *flagged* under the OLD rule and would be *validated* under the PI's §4 ruling. The bytes
are published either way; only the tier and the directory are in question.
