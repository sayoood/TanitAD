# PI decision queue, older items: 14 audited, 3 still want the PI

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind · **0 GPU**
**Source** `Project Steering/PI_DECISION_QUEUE.md` at tip `9e5f430` (read from the push mirror by
plumbing). **Appended** to it as `§C — audit 2026-09-19` (text: `raw/section_C.md`); no old item
was edited.

## The answer

| verdict | items |
|---|---|
| **CLOSED-BY-RULING** | **1**, **11**, **12**, **13** |
| **OVERTAKEN** | **3**, **5**, **7**, **9**, **10**, **14** |
| **NO PI RULING NEEDED** | **16** (default half-built; B2 restatement owed) · **v6F runbook** (a red test, not a decision) |
| ⭐ **STILL-LIVE — the PI's actual list** | **4** → fold into ITEM 26 · **6** (optional, newly relevant) · **v6F** only if it is to be revived |

Every verdict carries a quote, a `file:line` or a commit in §C and in `raw/evidence.json`.

## Four things the brief assumed that the evidence changed

1. ⛔ **The red test is not `test_nav_v6stack.py`** — that file passes, 17/17. The red one is
   `test_runbook_commands.py::test_every_runbook_launch_line_passes_the_trainers_own_preflight`,
   and the trainer's preflight refuses the v6F runbook lines for **two** reasons, not one:
   a missing `--nav-cond` (mandatory since the PI's 2026-08-30 directive) **and**
   `--horizons (1, 2, 4)`, whose heads 2 and 4 no loss consumes — they would train on
   exactly zero gradient. `raw/red_test_runbook_preflight.txt`.
2. ⭐ **Item 6 is MORE live than when it was filed, not less.** It was "available, deliberately
   not proposed" because nothing trained on the traffic-light colours. Now the tactical
   vocabulary carries `TRAFFIC_LIGHT_REACT_*`, the tactical-goal gate was widened on
   2026-09-18 so a refcv6 tactical arm receives those targets, and the PI's 2026-09-15 (c)
   has the tactical layer learn every behaviour from the labels. The spot-check now bounds
   something that will be trained.
3. ⚠️ **Item 16 is only half-built.** The detector logs `cos`, `ratio` and `proj`; but the
   restated criterion B2 exists only inside the queue item, in no pre-registration.
4. ⛔ **Item 4's repo exposes no clip ids — but two OTHER public repos do**, beside ITEM 26's.
   See below.

## Item 4 and ITEM 26: every `Sayood/*` repo (read-only HF audit)

48 repos: **28 public (901.3 GB)**, **20 private (176.9 GB)**. A file name "carries a raw id" if it
contains one of the 306,152 PhysicalAI clip ids or the 8-hex prefix of exactly one of them.

| repo | visibility | GB | files | names carrying a raw clip id |
|---|---|---|---|---|
| `tanitad-physicalai-w120-256x640cyl` | ⛔ **public** | 455.377 | 6,061 | **3,000** (ITEM 26) |
| `tanitad-ph0-aug120` | ⛔ **public** | 39.879 | 7,241 | ⭐ **6,833 — new finding** |
| `tanitad-flagship-v5f-w120` (a model repo) | ⛔ **public** | 31.691 | 97 | ⭐ **16 — new finding** |
| `tanitad-refc-v3` (item 4) | public | 2.142 | 12 | **0** |
| `tanitad-v7-training-corpus` | private | 74.269 | 10,349 | 10,200 (the corpus itself) |
| `tanitad-transfer-2026-08` | private | 0.697 | 24 | 20 |
| `tanitad-s2-lab` | private | 0.000 | 14 | 5 |
| the other 41 repos | — | — | — | 0 |

MEASURED · `raw/hf_repo_names_audit.json` (counts only; no id or prefix is recorded).
⇒ **ITEM 26's question — visibility and id exposure — covers THREE public repos, not one**,
**~527 GB** of public storage between them. Flipping all three private would take private
storage from 176.9 GB to ~704 GB, inside the 1 TB PRO allowance — **but add the 386.5 GB
corpus-cache push and it is ~1,090 GB, over the allowance and billed.** Same ordering trap as
ITEM 26 already records, now larger.

## Method, and what it cannot see

* Each item's own text was read at the tip, then every later ruling looked for in this queue,
  `Decisions/2026-09-15-pi-directives.md`, `GOALS_AND_CLAIMS.md`, `SPEC_REFCV6_V2.md`, the
  registry and the code the item names. Code claims were checked in the tip's code, not taken
  from prose (the trainer's refusal and acknowledgement lines were re-read by `grep -n`).
* Both v6 test files were **run** on D:, with `tanitad` confirmed to import from D:, and both
  test files confirmed identical to the tip before running.
* ⚠️ **Item 1 rests on the PI's words to the DataFlyWheel, not to the Master Mind.** They are
  first-hand in that session and banked verbatim in the v8 manifest; under the programme's
  relay rule the Master Mind may still record them as relayed. Nothing changes either way —
  the default is fully applied.

## Reproduce

`code/pdq_items.py <n…>` prints an item's own section compacted; `code/hf_repo_names_audit.py
<universe.json> <out.json>` repeats the HF audit (read-only; the universe is built from
`clip_index.parquet` and is deliberately not banked).


## Revision 2026-09-19 (after the PI's 2026-09-19 rulings, and item 17)

⚠️ **This section supersedes the verdict table above for items 3, 4, 5, 13 and 17. Nothing above
is edited.** The audit was written at `9e5f430`, before `537b028` recorded the PI's rulings of the
same day; the Master Mind's landing note under §C already answers item 4 with ITEM 26.

| item | revised verdict | evidence |
|---|---|---|
| **4** | **CLOSED-BY-RULING** | PI 2026-09-19: *"leave it as it is and make it with gated manual approval for access"*. Read back via the Hub API: **27 of 28** public repos are `gated: manual`, including all three whose file names carry clip ids and `tanitad-refc-v3`; no ungated public repo carries one (`raw/hf_public_gating.json`). |
| **3**, **5** | **CLOSED-BY-RULING — deferred**, not overtaken | PI 2026-09-19: *"the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality."* |
| **13** | **CLOSED-BY-RULING** — reinforced | 2026-09-19: *"5. we will do it later"* — the pod is parked. |
| **17** (added) | **CLOSED-BY-RULING** | PI 2026-09-17: *"Its hygiene and reproducibility."* — the record stays, `tools/clipid_scan.py` guards growth. |

**The guard, on tip `07541b7`:** the default UUID half reads **`NO-GROWTH`** (120 UUIDs in 27 files,
equal to the baseline). The prefix half is **unchanged since the ruling** — 530 → 530 against the
4,719-clip corpus list and 870 → 870 against the full index, **0 files grew** — but the baseline
stores **0 prefixes for every file**, so `--clips` against it reads red on every existing prefix.
Recording that floor is the guard owner's task. `raw/clipid_scan_tip_uuid_half.txt`,
`raw/clipid_growth_since_ruling.json`.

⛔ **Two corrections of my own, for the record.** (1) In the LAB-RUN-016 collection I presented
the tip-wide clip-id count as a new PI decision and redacted the register; the PI had closed that
question on 2026-09-17 (item 17), so the redaction is not landed. (2) While re-measuring, a direct
call into the scan module read **0** UUIDs where the CLI read 120 on the same tree — a broken
probe, discarded; every count above comes from the tool's own CLI.

⇒ **Still wanting the PI: item 6 (optional), and the v6F runbook only if v6F is to be revived.**

**Re-check, same evening:** the guard reads `NO-GROWTH` on `83d8ae3` and on `bbcfe4c` too — 120 UUIDs in 27 files; prefixes 530 / 870 as at the ruling; 0 files grew. `raw/clipid_guard_recheck.json`.

**⚠️ Superseded later the same day — the prefix floor is now recorded.** "The baseline stores 0 prefixes for every file" held on `07541b7` and no longer does: `tools/clipid_baseline.json` now records **530 prefixes in 53 files** against the **named** 4,719-clip v7 corpus list (sha256 `a48251e89c7a8603…`), and `--clips` with that list reads **NO-GROWTH** on `e0c31f2`. A run with any other list — the 306,152-clip index reads 870 — is refused rather than compared. `…/2026-09-19-clipid-prefix-floor/`.
