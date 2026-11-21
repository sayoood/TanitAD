# TanitAD program report — 2026-08-18 09:04 Europe/Berlin (07:04 UTC)

⛔ **THIS REPORT COULD NOT BE WRITTEN TO ITS CANONICAL PATH.** The Google-Drive `G:` mount has been
returning `Invalid request code` on **every content read** for ~1 h. `Project Steering/Reports/` is
unwritable and `.git` is unreachable. This file is banked on local disk and moves to
`Project Steering/Reports/2026-08-18-0904-program-report.md` when the mount returns.
**Nothing is lost: all 39 of the night's commits are pushed through `a603936`.**

⚠️ **Evidence-class note shaping the whole report:** Thor is a separate machine, so its numbers are
MEASURED now. Everything on `G:` — the SAM3 census, `git log`, the hub — is **INHERITED from the
last successful read** and labelled so. A mount-resident number is not restated as if measured.

---

## 1. Fresh measurements — v6F S-W 30k on Thor [MEASURED 07:04 UTC]

| | |
|---|---|
| step | **13,850 / 30,000 (46.2 %)** |
| step_s (cumulative) | **26.4611** |
| projected finish | **~4.95 days** — ESTIMATED, linear |
| trainer | **PID 25477 ALIVE** (matched on args, then kill -0) |
| snapshot daemon | **PID 42229 ALIVE** · 4 snapshots · next due ~14,000 |
| log freshness | **673 s** — normal at one line per ~22 min |
| disk free | **348 GB** |

gnorm-vs-loss health (n = 278 rows):

    loss   last 2.093 · run median 2.472 · last-40 median 2.081 · min 1.238 · max 19.530
    gnorm  last 679.3 · run median 371.3 · last-40 median 515.8 · min 15.1 · max 354,075.6
    rows above 2x run-median loss: 25 (none recent)

⇒ **HEALTHY.** The last loss sits BELOW the run median; gnorm is deeply mid-band against a historical
max of 354,075.

⚠️ **A correction this table respects:** step_s is a CUMULATIVE MEAN since process start, so it
cannot rise fast enough to trip a threshold (C112: at 40 s/step it needs 9 hours to cross a +5 %
bar). A true step_s_interval was built and is UNCOMMITTED on the failing mount. Until it lands this
figure is an ETA input, not a monitor.

---

## 2. Streams and knowledge transfer

⚠️ The `git log --since` half could not be run (mount). This is what each stream reported —
INHERITED from agent reports, not re-derived from the repo.

| stream | produced | state |
|---|---|---|
| Pooling ladder E-R1-0 + DINOv2 | **C104 — 40:1 pooling bottleneck REFUTED**; encoder is the constraint, 91x gap | integrated |
| C106 adversarial verification | **C109 — half right**; lead_gap dies, the 3.6x ratio withdrawn | integrated |
| planner_beats_cv re-drive | **C101 — CEM planner 35.8 % WORSE than CV at T1** | integrated |
| Ladder 3-seed re-run | **C107 — substantive count ZERO** on both routes | integrated |
| Alpamayo parity exclusion | **C112/C113 — 78.21 % buildable contamination; 6 of 40 val inside** | integrated |
| Build parity guard | **C118** — one ingest gate, six doors, 24 derived tests | integrated |
| Goal-provenance audit | **C120 — assert_isolation blind to forward information** | integrated |
| Per-stratum P7 | **C121 — the T3 verdict moves with the band edges** | integrated |
| F-7/F-8, F-9/F-11 cells | built, inert at default; C115 + C119 found | integrated |
| Thor stranded rescue | 117 files banked; **C111 token near-miss** | integrated |
| **F-10 / F-14 cells** | F-10 built zero-parameter; **F-14 REFUSED with the arithmetic** | **UNCOMMITTED — failing mount** |
| **Credential scanner** | 6,201 files / 0 blocking; **corrects C117** | **UNCOMMITTED — failing mount** |
| SSL world-model literature | running | in flight |

---

## 3. SAM3 perception backfill census

⛔ **NOT MEASURED THIS MORNING — the manifest is on the failing mount.** Per C77 a record counts as
complete only if it carries the liveness control AND has zero error entries — a CONTENT read. A
remembered number is not a census.

INHERITED, last successful read (2026-08-17): 201/201 clips, UNIFIED/HOMOGENEOUS true, residual
empty, one floor (0.25), one schema (v2), perception_engine_mixed false, zero dead-control failures.

⛔ **Its publication status was WRONG in my 2026-08-17 report (C117).** pushed_to_hf is FALSE on both
manifests; only the older 115-clip leg is on HF, in a PRIVATE repo. ⇒ **The unified 201-clip corpus
exists on ONE DISK — the disk currently faulting.**

---

## 4. Position vs the four edges

### PLANNING — WEAK, now precisely localised
C101 (T1, CI-separated): the CEM planner is **35.8 % worse than constant velocity**, and loses on
LONGITUDINAL, the family its own cost targets. Operative-under-true-actions BEATS CV (-0.3151 m)
⇒ **the loss is in the ACTION SEARCH, not the world model.**

### EFFICIENCY (self-supervised WM) — DOWNGRADED to weak
C104: removing the 40:1 pool moves r2 by <= 0.0002 (CI contains zero, five seeds), while
dinov2-base at **86 M vs our 87.3 M** reads lead_gap **0.44997 vs 0.00496** through the same pool on
the same windows ⇒ **the ENCODER/OBJECTIVE is the constraint, 91x.** C107: at three seeds the
ladder's substantive count is ZERO.

### SAFETY / SELF-KNOWLEDGE — STRONG, still the best edge
32 retraction classes this session (C90-C121). The pattern: **every verification level trusted proved
insufficient at the next** — presence, md5, import, then CURRENCY; negative, positive, then
TRIVIAL-PROXY controls; one seed then THREE; a control per study then PER ARM; and a count is a
claim about the FILTER.

### DATA EFFICIENCY — DOWNGRADED
C112/C113: the buildable Alpamayo eval split is **78.21 %** contaminated, and **6 of the 40 canonical
val episodes** sit inside the record set. Blast radius zero TODAY, trigger already scheduled. C118
now gates six build doors and REFUSES. But the unified corpus is unpublished and single-disk (C117).

---

## 5. Ordered next steps

1. ⛔ **Restore the Drive mount** — blocks every repo action.
2. **Land F-10/F-14 and the credential scanner** (both complete, both uncommitted).
3. **Re-run the committed-history secret audit** — currently SCAN UNUSABLE, 0 of 5,487 blobs read,
   i.e. **unanswered, not clean**.
4. **At 30k (~5 days):** P1/P3/P6 battery, then D1 re-run (9 fits), then S-T launch WITH --v2-lru 64.
5. **Encoder experiments (E-XENC-1/2)** now outrank R1/R2 — they are what C104 indicts.
6. **Ship pending code to Thor at the S-T boundary** via launch_closure_audit.py --ship.

---

## 6. Decisions required from Sayed — each with a default

| # | decision | default if you say nothing |
|---|---|---|
| D-a | ⛔ **Rotate the HF token** — plaintext on Thor at ~/rq_out/logs/contention.log:11, WRITE access to Sayood/ | **Rotate.** No default makes an exposed credential safe. |
| D-b | **Push the unified 201-clip corpus off one disk** — and that disk is faulting | **Push, gated-public** — you granted this pattern on 2026-08-17. |
| D-c | **Sub-300M**: the live run is 336,542,025 params, **12.2 % over** | **Restate the claim.** Rescoping mid-run costs the 30k. |
| D-d | **Pre-register the T3 strata** — the gate can be made to pass OR fail by band choice (C121) | **Edge-free LEAD-vs-NO_LEAD** as primary; 3-band reported, never gating. |
| D-e | **4.70 GB val-cache download** to close planner_beats_cv open-loop | **Defer** — the substantive question is answered at T1 (C101). |
| D-f | **DINOv3 licence** (gated manual, 403 at three probes) | **Accept** — unblocks the top architecture item. |
| D-g | **4,472-clip build: separate corpus or parity extension?** | **No default — no guard can make this call.** |

---

## 7. Incidents — honestly

* ⛔ **The Drive mount failed mid-session**, and its probe LIES THE WAY df DOES: ls and stat succeed
  while content reads fail. My first rescue trusted stat; all five file copies failed.
* ⛔ **I reported the perception corpus as "published gated-public". It is on one disk** (C117) —
  wrong artifact AND wrong visibility, because "done" was inherited across a superseding artifact.
* ⛔ **I reported "no credential scanner exists" from three probes. One has existed since
  2026-07-25.** All three asked "is a scanner PRODUCT installed?" — **three probes of the same shape
  are one probe.**
* ⛔ **C111 was misdiagnosed**: the old scanner DOES catch the leaked token's exact shape. It leaked
  because NOTHING CALLED IT — and a 2026-07-26 harvest already recorded safe_commit.py as
  "imported_by: Nothing" with a one-line fix. **C111 is the invoice for that unmerged line, 24 days
  later.**
* ⚠️ **A false loss alarm** on Thor — a single sample compared against a LAST-40 median rather than
  the run median.
* ⚠️ **Wrong filename in a commit** (goal_admissibility.py for goal_provenance.py) — guessing from
  the topic instead of reading the manifest.

---

## 8. Evidence classes

Every Thor number in section 1 is **MEASURED** (07:04 UTC, the machine's own log). The finish
projection is **ESTIMATED**. Sections 2 and 3 are **INHERITED** and labelled in place.
Retraction-class findings are MEASURED by their originating stream with artifact paths in
RETRACTION_LOG.md.

**Tier stamps:** C101 is **T1** (primary). Every probe in section 4's EFFICIENCY paragraph is
**T0-DIAGNOSTIC** — a WM diagnostic, never driving performance.
