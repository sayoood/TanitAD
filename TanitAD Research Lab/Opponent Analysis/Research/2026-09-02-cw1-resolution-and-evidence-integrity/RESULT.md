# RESULT — E-OPP-CW1-1

`2026-09-02 · Research Lab · Opponent Analysis · 0 GPU · raw/`

**Tier: N/A** — opponent-claim adjudication and artifact integrity, not a model
read. All competitor numbers are **PUBLISHED (primary, banked)**; every one below
was read from a PDF we hold, never from an aggregator.

**Headline.** ⭐ **CW-1 is RESOLVED from the primary's own text** — DriveFuture's
ablation rows are run *"without GTRS-Dense scorer"* while the 55.5 headline
includes it. Getting there required repairing the evidence base: the banked
DriveFuture PDF was **truncated to 14.5 %** of the real file, **and
`kb_add.py --verify` reported the whole Library healthy anyway** — before *and
after* the repair, because it re-hashes rather than re-reads. Along the way the
leaderboard's DrivoR line turns out to be wrong on two counts, and the efficiency
wedge in backlog row 32 needs restating.

---

## F1 — ⭐⭐ CW-1 RESOLVES: THE ABLATIONS RUN WITHOUT THE SCORER

**PUBLISHED (primary, banked: `2605.09701`, repaired copy, 24 pp).** The
reconciling sentence is the ablation table's own caption:

> *"All ablations are conducted **without GTRS-Dense scorer [49]**."*

⇒ The 30.9–34.6 rows and the 55.5 headline are **different configurations**, and
the difference is named in the paper. CW-1's premise — *"cannot be reconciled with
its own navhard ablation rows from the text"* — was true **of our copy**, which was
missing 85.5 % of the document including that table.

The paper's headline, verbatim: *"reaching 55.5 EPDMS on NAVSIM-v2 navhard, 89.9
EPDMS on NAVSIM-v2 navtest, and 90.7 PDMS on NAVSIM-v1 navtest."*

### ⭐ Independently corroborated by a second banked primary

`2606.07170` (TOAD, valeo.ai) evaluates six planners on the same benchmark and
reports DriveFuture at **55.5** in its Table 6 — *"outperforming the strongest
learned method (DriveFuture [41], see in Tab. 6, 55.5)"*. ⇒ the level is
reproduced by an unrelated group.

### ⇒ Recommendation: NARROW the fence, do not simply lift it

⛔ **The Lab measures; it does not un-fence an opponent claim.** Escalated. The
recommendation is that a DriveFuture navhard **level** becomes admissible **only
with its setting attached** — `NAVSIM-v2 navhard, with GTRS-Dense scorer` — since
the whole defect was a level travelling without one.

## F2 — ⭐⭐ AND THE RESOLUTION CARRIES A FINDING WORTH MORE THAN THE FENCE

The same caption makes DriveFuture's own ablation an **isolation of the
world-model contribution**, because it strips the scorer:

| configuration | EPDMS (navhard) |
|---|---|
| future frames only | **30.9** |
| + implicit future constraint (the paper's mechanism) | **34.6** |
| + MSE future-latent supervision instead | 32.1 |
| − kinematic/GT guidance | 32.0 |
| ⭐ **headline, WITH GTRS-Dense scorer** | **55.5** |

⇒ **The future-aware latent world model is worth +3.7 EPDMS (30.9 → 34.6) in the
paper's own isolation. The remaining +20.9 to the headline comes with the dense
scorer.** Of the 24.6-point span between the weakest ablation and the headline,
the world-model mechanism accounts for **~15 %**.

⭐ **Why this matters to us:** the strongest *published latent-world-model* result
on navhard is dominated by a **scoring/selection** component, not by the world
model. That is the same shape as our own record (the tactical/selection surface
carrying more than the predictor) and it is a **PUBLISHED** datapoint we can cite
for it. ⚠️ It is also a caution: a headline labelled "latent world model SOTA" is
not a measurement of latent world modelling.

## F3 — ⛔ `LEADERBOARD.md:1036` IS WRONG ON TWO COUNTS, MEASURED AGAINST THE PRIMARY

`Benchmarks & Eval/LEADERBOARD.md:1036-1037` reads:

> *"… DrivoR 56.3 EPDMS **navhard**, arXiv 2606.07170 · DriveFuture 55.5 EPDMS
> **navhard**, arXiv 2605.09701 …"*

`2606.07170` Table 3, verbatim: `DrivoR [8] 54.6` · `+ TOAD 56.3 (↑3.1%)`; Table 7
names the arm `DrivoR (ViT-S) [8] … 54.6`.

| the line says | the primary says |
|---|---|
| DrivoR = **56.3** | ⛔ **56.3 is DrivoR + TOAD** — a Cross-Entropy-Method **test-time search** bolted onto a frozen DrivoR. **Bare DrivoR = 54.6.** |
| DrivoR, **arXiv 2606.07170** | ⛔ `2606.07170` is **TOAD's** id. DrivoR is **`2601.05083`** ("Driving on Registers"). The line pairs a model name with a different paper's id *and* that paper's augmented number. |

⭐ **The correct setting was already banked and then dropped downstream:** the
Library's own note on `2606.07170` reads *"DrivoR 56.3 EPDMS **w/ TTO**"*.
⇒ Same error class as CW-1 — **a level quoted across settings** — sitting one line
below CW-1's own subject, and unflagged.

⭐ **DrivoR 54.6 is confirmed by two independent primaries:** TOAD's Table 3/7, and
DriveFuture's own Figure 1(c) (*"DrivoR CVPR'26 54.6"*).

## F4 — ⚠️ THE EFFICIENCY WEDGE (BACKLOG ROW 32) IS BUILT ON THE WRONG LEADER

Row 32 states *"DrivoR leads NavSim camera-only at ~40 M params."* From TOAD's
Table 6 (NAVSIM-v2 navhard-two-stage, EPDMS):

```
PDM-Closed        56.6   ⚠️ PRIVILEGED — uses ground-truth perception
DrivoR + TOAD     56.3   test-time search on a frozen planner
DriveFuture       55.5   ⭐ the strongest LEARNED method (TOAD's own words)
DrivoR (ViT-S)    54.6
ZTRS              48.1 · GTRS 45.4 · Hydra-MDP 40.9 · RAP-DINO 39.6 · iPad 34.7
```

⇒ **Bare DrivoR is fourth, behind DriveFuture.** It leads only *with* test-time
search, which is inference compute rather than parameters — so the 56.3 is not a
parameter-efficiency datapoint at all.

⛔ **What I did NOT verify: the ~40 M parameter count.** That needs `2601.05083`,
which is banked and readable (13.5 MB, verdict OK) but was **not read this pass**.
⇒ **The parameter half of the wedge remains UNVERIFIED and must not be quoted.**
Proposed as backlog row **L-15**.

## F5 — ⛔⛔ `kb_add.py --verify` CANNOT DETECT A CORRUPT PRIMARY, AND SAID SO TWICE

**MEASURED** (ours; `raw/library_integrity.json`, byte-level checks over all 310
banked entries):

```
TALLY {'OK': 309, 'TRUNCATED_NO_EOF': 1}
```

The one failure is **`2605.09701` — the exact paper CW-1 needed**: 1,373,148 B
banked against a true **9,453,407 B** (**14.5 %**), **no `%%EOF` anywhere**, pypdf
`Stream has ended unexpectedly`.

⭐ **The structural finding, and it is the reason this is in a package rather than
a bug report.** `kb_add.py --verify` re-hashes each file against the hash recorded
**at bank time**. The file was already truncated when it was hashed, so the hash
matched — and it reported:

```
before the repair : verified 310 entries, 0 orphan(s), 0 problem(s)
after  the repair : verified 310 entries, 0 orphan(s), 0 problem(s)
```

⇒ **The same verdict for a corrupt Library and a healthy one.** `--verify` proves a
file has not *changed*; it cannot prove a file was ever *complete*. This is the
`kb_add` class already on the record — *"the tool banked garbage on its first run
and reported success"* — recurring through a different door, and it is precisely
the `LAB_BACKLOG` row 36 hazard, now measured rather than predicted.

### ✅ REPAIRED THIS PASS

Refetched from arXiv (9,453,407 B, `%%EOF` present, 24 pages parse, page 1 yields
2,990 chars), overwritten with a sha256 read-back, and the index entry's `bytes` /
`sha256` updated atomically with the old hash recorded in its note. Post-repair:
`--verify` → 310/0/0 with the **correct** hash, and the integrity check reads
`verdict: OK, n_pages: 24`. A backup of the pre-repair index is at
`raw/library.json.pre-repair.bak`.

⚠️ **Honest framing: the Library is in good shape.** 309 of 310 are fine — this is
**not** a rot narrative, and it should not be quoted as one. What it is: a 0.3 %
corruption rate that the guard structurally cannot see, which happened to land on
an open correction-watch's primary.

## F6 — ⭐ TOAD IS EXTERNAL EVIDENCE FOR OUR PLANNING LINE, AND IT CARRIES A WARNING FOR I-1/L-3

`2606.07170` attaches one frozen scorer to six frozen planners and improves all
six with **no retraining**: **+2.3 % (ZTRS) to +43.6 % (iPad)** EPDMS on NAVSIM-v2.
That is independent, primary support for the programme's test-time latent-planning
direction (injected row **I-1**, backlog **L-3**).

⛔ **But the warning is sharper than the support, and it lands on L-3's design.**
TOAD, verbatim:

> *"simply re-ranking the base planner's proposals with the scorer can even hurt
> performance, whereas TOAD improves it."*

**Re-ranking is not search.** I-1/L-3 compare a learned terminal value head against
**fan-scoring** — and fan-scoring *is* re-ranking. ⇒ an L-3 that pits a value head
against fan-scoring alone is comparing two members of the family TOAD reports as
the weaker one, and could conclude "value heads don't help" from a design that
never contained the lever. **L-3 should carry a search arm** (CEM over proposals,
warm-started from them) or say why not.

⚠️ **And a caution about reading our own future results:** *"gains are largest for
lower-performing planners"* (+43.6 % on the weakest, +3.1 % on the strongest). Our
arms are far from SOTA, so a large test-time-search gain on them would be the
**expected** result for a weak base planner — **not** evidence of a strong world
model.

---

## What this changes

1. **`Frontier Scan/LEDGER_A1_world_models.md:60-63` (CW-1)** — resolvable now;
   recommend narrowing to *"admissible with its setting"* rather than lifting.
   ⛔ PI/Master Mind call, not the Lab's.
2. **`Benchmarks & Eval/LEADERBOARD.md:1036`** — two corrections (56.3 → 54.6 for
   bare DrivoR; `2606.07170` is TOAD, DrivoR is `2601.05083`).
3. **`LAB_BACKLOG.md` row 32** — the wedge's premise names the wrong leader; the
   ~40 M parameter figure is unverified.
4. **`LAB_BACKLOG.md` row 36** (`kb_add.py` hardening) — add a **completeness**
   check at bank time (`%%EOF` + a parse), because hash-verification structurally
   cannot supply one.
5. **Backlog rows I-1 / L-3** — add a search arm; re-ranking is the weaker family.

## Limits, stated

* DrivoR's **parameter count is unverified** (F4). `2601.05083` is banked and
  readable but unread this pass.
* All EPDMS numbers are **NAVSIM-v2 navhard / navhard-two-stage**. ⚠️ The
  programme's own NAVSIM date fence still applies: the scoring basis moved
  2025-04-28 and 2025-09-29, and comparability across those dates is **unchecked**
  for every number above. They are quoted here as *opponent-internal* comparisons
  (all six planners scored by TOAD in one table), which is the setting in which
  they are safest.
* F2's "~15 %" is arithmetic on the paper's own ablation table. The ablations may
  differ in ways beyond the scorer; the caption names the scorer, and that is what
  is claimed.
* ⛔ **No comparison to any TanitAD number is made or implied.** We hold no navhard
  EPDMS row (`LEADERBOARD.md` records ours as *"NOT COMPUTABLE TODAY"*).
